"""Fast visible tracking continues while a learned worker refines correspondence.

A late answer belongs to its older capture. It reaches the current view
only through retained actual pictures and fresh forward/reverse matching.
No learned worker can queue a second operation behind its unfinished one.
"""
from __future__ import annotations

import copy
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.perception.points_before_the_picture_is_reduced import (
    PointsBeforeReduction,
    _colours,
)
from core.perception.points_that_keep_their_motion import SeenPoint, _grey, _pyramid
from core.runtime.executors import submit_blocking_io
from core.runtime.lockdep import checked_thread_semaphore
from core.runtime.optional_compute import OptionalComputeBusy
from core.verify import invariant

_SLOTS = checked_thread_semaphore('perception.learned_point_corrections', 1)
_LEARNED_NAMES = 1 << 48


@dataclass
class _Picture:
    source: np.ndarray
    working: np.ndarray
    at: float
    generation: int


@dataclass
class _Correction:
    future: Any
    pictures: tuple[_Picture, ...]
    reset: bool = False
    stage: str = 'learned'


@dataclass(frozen=True)
class _Refinement:
    points: tuple[SeenPoint, ...]
    picture: _Picture


class _VisibleTransport(PointsBeforeReduction):
    """Follow only the supplied named marks; fresh unrelated corners receive no credit."""

    def _spread(self, points: list[SeenPoint], scale: tuple) -> tuple[SeenPoint, ...]:
        return tuple(point for point in points if point.number in self._allowed)[:self.limit]


def _native_picture(source: np.ndarray, budget: int) -> np.ndarray:
    factor = min(1., math.sqrt(budget / (source.shape[0] * source.shape[1])))
    if factor >= 1.:
        return source
    from core.perception.picture_arithmetic import shrink

    return shrink(source, int(source.shape[1] * factor), int(source.shape[0] * factor))


def visible_transport(points: tuple[SeenPoint, ...], pictures: list[_Picture],
                      budget: int) -> _VisibleTransport | None:
    """An older learned position is a seed; each intervening displacement needs actual pixels."""
    if not points or len(pictures) < 2:
        return None
    anchor = pictures[0]
    if any(point.at != anchor.at for point in points):
        return None
    native = _native_picture(anchor.source, budget)
    scale = anchor.working.shape[1] / native.shape[1], anchor.working.shape[0] / native.shape[0]
    transport = _VisibleTransport(limit=len(points), pixel_budget=budget)
    transport._allowed = {point.number for point in points}
    transport._native_points = tuple(SeenPoint(p.number, p.x / scale[0], p.y / scale[1], p.x / scale[0], p.y / scale[1],
        p.born, anchor.at, anchor.at, p.error, 1) for p in points)
    transport._number = max(transport._allowed) + 1
    transport._geometry = anchor.source.shape[:2], native.shape[:2], anchor.working.shape[:2]
    transport._image, transport._colours, transport._at = _pyramid(_grey(native)), _colours(native), anchor.at
    for picture in pictures[1:]:
        if picture.generation != anchor.generation:
            return None
        transport.see_source(picture.source, picture.working, picture.at)
        if not transport.points:
            return None
    return transport


def _advance_transport(transport: _VisibleTransport | None, pictures: list[_Picture]) -> _VisibleTransport | None:
    if transport is None or not pictures or transport._at != pictures[0].at:
        return None
    for picture in pictures[1:]:
        if picture.generation != pictures[0].generation:
            return None
        transport.see_source(picture.source, picture.working, picture.at)
        if not transport.points:
            return None
    return transport


def _retain_visible_marks(previous: _VisibleTransport | None, current: _VisibleTransport,
                          limit: int) -> _VisibleTransport:
    """A refinement may add marks without discarding independently visible moving texture."""
    if previous is None or previous._at != current._at or previous._geometry != current._geometry:
        return current
    scale = current._geometry[2][1] / current._geometry[1][1], current._geometry[2][0] / current._geometry[1][0]
    retained = sorted(previous._native_points, key=lambda p: (
        -math.hypot((p.x - p.old_x) * scale[0], (p.y - p.old_y) * scale[1]), -p.pictures, p.error, p.number))[:limit]
    names = {p.number for p in retained}
    for point in current._native_points:
        if len(retained) >= limit:
            break
        if point.number not in names and all(math.hypot((point.x - p.x) * scale[0], (point.y - p.y) * scale[1]) >= .5
                                            for p in retained):
            retained.append(point)
            names.add(point.number)
    current.limit, current._allowed = limit, names
    current._native_points = tuple(retained)
    current._number = max(names, default=-1) + 1
    current.points = tuple(SeenPoint(p.number, p.x * scale[0], p.y * scale[1], p.old_x * scale[0], p.old_y * scale[1],
        p.born, p.before, p.at, p.error, p.pictures) for p in retained)
    return current


def _completed(future: Any, slots: Any) -> None:
    try:
        if not future.cancelled():
            future.exception()
    finally:
        slots.release()


class CorrectionsWhilePointsMove(PointsBeforeReduction):
    """The observation worker owns fast pixels; a second worker owns the learned state."""

    def __init__(self, teacher_factory: Any, *, limit: int = 192, pixel_budget: int = 256_000,
                 retained_pictures: int = 32, sequence_required: bool = False,
                 batch_frames: int = 32) -> None:
        super().__init__(limit=limit, pixel_budget=pixel_budget)
        if type(retained_pictures) is not int or not 2 <= retained_pictures <= 32:
            raise ValueError('point corrections need bounded retained pictures')
        if (type(sequence_required) is not bool or type(batch_frames) is not int
                or not 1 <= batch_frames <= 32):
            raise ValueError('point sequences need a bounded capture batch')
        self._factory, self._teacher = teacher_factory, None
        self._sequence_required, self._batch_frames = sequence_required, batch_frames
        self._submitted_at: float | None = None
        self._teacher_generation: int | None = None
        self._generation = 0
        self._failed_generation: int | None = None
        self._retained: deque[_Picture] = deque(maxlen=retained_pictures)
        self._operation: _Correction | None = None
        self._learned: tuple[SeenPoint, ...] = ()
        self._transport: _VisibleTransport | None = None
        self.correction_status = 'not started'
        self._closed = False
        self._deferred_until = 0.

    def clear(self) -> None:
        super().clear()
        self._generation += 1
        self._retained.clear()
        self._learned = ()
        self._transport = None
        self._failed_generation = None
        self._submitted_at = None
        self._deferred_until = 0.
        self.correction_status = 'reset'

    def _refine(self, pictures: tuple[_Picture, ...], reset: bool) -> _Refinement:
        if self._teacher is None:
            self._teacher = self._factory()
        if reset or self._teacher_generation != pictures[0].generation:
            self._teacher.clear()
            self._teacher_generation = pictures[0].generation
        latest = _Refinement((), pictures[-1])
        for picture in pictures:
            points = copy.deepcopy(self._teacher.see_source(picture.source, picture.working, picture.at))
            if points:
                latest = _Refinement(points, picture)
        return latest

    def _receive(self) -> None:
        operation = self._operation
        if operation is None or not operation.future.done():
            return
        if operation.pictures[0].generation != self._generation:
            self._operation = None
            self.correction_status = 'correction expired'
            return
        try:
            result = operation.future.result()
            if operation.stage == 'learned':
                if not isinstance(result, _Refinement) or not any(result.picture is p for p in operation.pictures):
                    raise ValueError('a point refinement must name its actual submitted capture')
                anchor = result.picture
            else:
                anchor = operation.pictures[-1]
                if result is not None and (not isinstance(result, _VisibleTransport) or result._at != anchor.at):
                    raise ValueError('visible transport must answer its latest submitted capture')
            pictures = list(self._retained)
            index = next((i for i, p in enumerate(pictures) if p is anchor), None)
            if index is None:
                self._operation = None
                self.correction_status = 'correction expired'
                return
            pictures = pictures[index:]
            empty = not result.points if operation.stage == 'learned' else result is None
            if not empty and len(pictures) > 3:
                self._transport_in_background(operation, result, pictures)
                return
            self._operation = None
            transported = (visible_transport(result.points, pictures, self.pixel_budget)
                           if operation.stage == 'learned' else _advance_transport(result, pictures))
            if transported is not None:
                self._transport = _retain_visible_marks(self._transport, transported, min(192, self.limit))
            self.correction_status = 'current visible correction' if transported is not None else 'no visible correspondence'
        except OptionalComputeBusy as why:
            self._operation = None
            self.correction_status = f'correction deferred: {why}'
            self._deferred_until = time.monotonic() + .5
        except Exception as why:
            from core.runtime.errors import record_degradation

            self._operation = None
            record_degradation('learned_point_correction', why, action='discarded the unavailable correction')
            self.correction_status = 'correction unavailable'
            self._failed_generation = self._generation

    def _transport_in_background(self, operation: _Correction, result: Any, pictures: list[_Picture]) -> None:
        """Catch up a late answer without turning retained pictures into a long current-frame operation."""
        slots = _SLOTS
        if not slots.acquire(blocking=False):
            self.correction_status = 'waiting for visible transport'
            return
        try:
            if operation.stage == 'learned':
                future = submit_blocking_io(visible_transport, result.points, pictures, self.pixel_budget,
                                            label='learned_point_visible_transport')
            else:
                future = submit_blocking_io(_advance_transport, result, pictures, label='learned_point_visible_transport')
        except BaseException:
            slots.release()
            raise
        future.add_done_callback(lambda done: _completed(done, slots))
        self._operation = _Correction(future, tuple(pictures), stage='transport')
        self.correction_status = 'following the intervening pictures'

    def _submit(self, picture: _Picture) -> None:
        slots = _SLOTS
        if (self._closed or time.monotonic() < self._deferred_until or self._failed_generation == self._generation
                or self._operation is not None or not slots.acquire(blocking=False)):
            return
        pictures, reset = self._batch(picture)
        if not pictures:
            slots.release()
            return
        try:
            future = submit_blocking_io(self._refine, pictures, reset, label='learned_point_correction')
        except BaseException:
            slots.release()
            raise
        future.add_done_callback(lambda done: _completed(done, slots))
        self._operation = _Correction(future, pictures, reset)
        self._submitted_at = pictures[-1].at

    def _batch(self, picture: _Picture) -> tuple[tuple[_Picture, ...], bool]:
        """Windowed trackers receive actual consecutive captures; an overrun starts new queries."""
        if not self._sequence_required:
            return (picture,), False
        retained = tuple(p for p in self._retained if p.generation == picture.generation)
        fresh = tuple(p for p in retained if self._submitted_at is None or p.at > self._submitted_at)
        anchor_kept = self._submitted_at is None or any(p.at == self._submitted_at for p in retained)
        reset = not anchor_kept or len(fresh) > self._batch_frames
        return fresh[-self._batch_frames:], reset

    def see_source(self, source: np.ndarray, working: np.ndarray, at: float) -> tuple[SeenPoint, ...]:
        fast = super().see_source(source, working, at)
        if self._transport is not None:
            self._transport.see_source(source, working, at)
        picture = _Picture(np.array(_native_picture(source, self.pixel_budget), copy=True),
                           np.array(working, copy=True), at, self._generation)
        self._retained.append(picture)
        self._receive()
        current = self._current_learned(at)
        # Two names on one measured mark do not become two pieces of support.
        self.points = (*current, *(point for point in fast if all(math.hypot(point.x - p.x, point.y - p.y) >= .5
                                                                 for p in current)))
        self._remember_points(replacing_current=True)
        self._submit(picture)
        return self.points

    def close(self) -> None:
        """The pending worker closes its teacher after its last operation finishes."""
        self._closed = True
        operation = self._operation
        if operation is not None and not operation.future.done():
            operation.future.add_done_callback(lambda _done: self._close_teacher())
        else:
            self._close_teacher()

    def _close_teacher(self) -> None:
        teacher, self._teacher = self._teacher, None
        if teacher is not None:
            close = getattr(teacher, 'close', None)
            if callable(close):
                close()
        self._retained.clear()
        self._transport = None

    def _current_learned(self, at: float) -> tuple[SeenPoint, ...]:
        carried = self._transport.points if self._transport is not None else ()
        return tuple(SeenPoint(p.number + _LEARNED_NAMES, p.x, p.y, p.old_x, p.old_y,
            p.born, p.before, p.at, p.error, p.pictures) for p in carried if p.at == at)


@invariant('perception.late_points_need_visible_transport', scope='perception',
           owner='core/perception/learned_tracking/corrections_while_points_keep_moving.py', observational=False)
def _visible_correction_invariant() -> tuple:
    pixels = np.full((48, 64, 3), 25, np.uint8)
    pixels[12:36, 16:40] = np.random.default_rng(72).integers(30, 230, (24, 24, 3), dtype=np.uint8)
    point = SeenPoint(1, 22., 18., 22., 18., 1., 1., 1., .01, 1)
    frames = [_Picture(pixels, pixels, 1., 0), _Picture(pixels, pixels, 1.1, 0)]
    current = visible_transport((point,), frames, 256_000)
    assert current is not None and current.points[0].at == 1.1 and current.points[0].before == 1.
    frames[1].generation = 1
    assert visible_transport((point,), frames, 256_000) is None
    return ()


@invariant('perception.grouped_points_use_actual_capture_sequences', scope='perception',
           owner='core/perception/learned_tracking/corrections_while_points_keep_moving.py', observational=False)
def _capture_batch_invariant() -> tuple:
    points = CorrectionsWhilePointsMove(lambda: None, sequence_required=True, batch_frames=2)
    pixels = np.zeros((32, 32, 3), np.uint8)
    pictures = tuple(_Picture(pixels, pixels, at, 0) for at in (1., 1.1, 1.2, 1.3))
    points._retained.extend(pictures[:3])
    points._submitted_at = 1.
    batch, reset = points._batch(pictures[2])
    assert batch == pictures[1:3] and not reset
    points._retained.append(pictures[3])
    batch, reset = points._batch(pictures[3])
    assert batch == pictures[2:] and reset
    return ()


@invariant('perception.refinement_keeps_current_visible_marks', scope='perception',
           owner='core/perception/learned_tracking/corrections_while_points_keep_moving.py', observational=False)
def _visible_mark_retention_invariant() -> tuple:
    pixels = np.random.default_rng(72).integers(30, 230, (48, 64, 3), dtype=np.uint8)
    frames = [_Picture(pixels, pixels, 1., 0), _Picture(pixels, pixels, 1.1, 0)]
    marks = tuple(SeenPoint(i, float(x), float(y), float(x), float(y), 1., 1., 1., .01, 1)
                  for i, (x, y) in enumerate(((18, 18), (30, 18), (18, 30), (30, 30))))
    old = visible_transport(marks, frames, 256_000)
    fresh = visible_transport(tuple(SeenPoint(p.number + 10, p.x, p.y, p.old_x, p.old_y,
        p.born, p.before, p.at, p.error, p.pictures) for p in marks), frames, 256_000)
    assert old is not None and fresh is not None
    combined = _retain_visible_marks(old, fresh, 8)
    assert {p.number for p in combined.points} == {p.number for p in old.points}
    assert all(p.at == 1.1 and p.before == 1. for p in combined.points)
    return ()
