"""Collect a finite pixel experiment through the caller's owned input lane.

The caller supplies an exclusive input lease and routes every physical input
through PixelExperimentInputs while collection runs. The frame getter is the
caller's CanvasFrames.look binding. Missing capture or delivery evidence ends
the experiment; this collector supplies no body identity or steering policy.
"""
from __future__ import annotations

import asyncio
import logging
import math
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from core.agency.inputs_while_measuring_control import (
    HandsWithControlReceipts,
    _control_delivery_lock,
    _hold_control,
    release_control_if_owned,
)
from core.agency.pixels_inside_control_trials import (
    FrozenPixelFrame,
    PixelPress,
    RandomizedPixelExperiment,
    freeze_frame,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PixelCollectionResult:
    status: str
    trials: int
    release_confirmed: bool
    release_problem: str | None = None


class _LostEvidenceError(Exception):
    pass


class PixelExperimentInputs(HandsWithControlReceipts):
    """Keep the caller's lease, acknowledged key, and unrelated-input history."""

    def __init__(self, hands: Any, run: Any, hers: Any, *, owns: Callable[[], bool]) -> None:
        if not callable(owns):
            raise TypeError("pixel collection requires the caller's input lease check")
        if isinstance(hands, HandsWithControlReceipts):
            if hands.run is not run or hands.hers is not hers:
                raise ValueError("physical input receipts must belong to this caller")
            hands = hands.hands
        super().__init__(hands, run, hers)
        self.owns = owns
        self._interference = 0
        self._collector: object | None = None
        self._pending: asyncio.Task | None = None
        self._pending_key = ''
        self._pending_previous: tuple = ()
        self._pending_began = 0.
        self._dispatch_attempted = False
        self._pending_release: asyncio.Task | None = None
        self._release_previous: tuple = ()
        self._cleanup: asyncio.Task | None = None
        self._owned_press: tuple[str, float] | None = None
        self.release_problem: str | None = None

    @property
    def pending_dispatch(self) -> tuple[str, float] | None:
        """An attempted dispatch remains owned until its acknowledgement is resolved."""
        return (self._pending_key, self._pending_began) if self._pending is not None and self._dispatch_attempted else None

    def _other_input(self, name: str, at: float) -> None:
        self._interference += 1
        super()._other_input(name, at)

    def _idle(self, *, dispatching: bool = False) -> bool:
        deadline = getattr(self.run, '_trial_input_deadline', None)
        return (self.owns() is True and not self.run.held
                and (self._pending is None or dispatching) and self._pending_release is None
                and self._owned_press is None and self.release_problem is None
                and not getattr(self.run, 'letting_go', ())
                and not getattr(getattr(self.run, 'charging', None), 'key', '')
                and getattr(self.run, '_prepared_input_set', None) is None
                and getattr(self.run, '_reflex_input', None) is None
                and (deadline is None or deadline.done()))

    def _receipts(self) -> tuple:
        return tuple(getattr(self.run, '_input_clock_receipts', ()))

    async def _deliver(self, key: str, ack_s: float) -> tuple[float, float]:
        previous = self._receipts()
        self._pending_key, self._pending_previous = key, previous
        self._pending_began = time.monotonic()
        self._dispatch_attempted = False
        self._pending = asyncio.create_task(self._press_if_neutral(key, previous, self._interference))
        await asyncio.wait_for(asyncio.shield(self._pending), ack_s)
        delivery = self._new_receipt(previous, key)
        if (delivery[0] < self._pending_began or self.run.held != key
                or self.run.held_since != delivery[1]):
            raise _LostEvidenceError("input_owner_changed")
        self._owned_press = key, delivery[1]
        self._pending = None
        self._dispatch_attempted = False
        return delivery

    async def _press_if_neutral(self, key: str, previous: tuple, interference: int) -> None:
        async with _control_delivery_lock(self.run):
            if not self._idle(dispatching=True) or self._receipts() != previous or self._interference != interference:
                raise _LostEvidenceError("neutral_owner_changed")
            self._dispatch_attempted = True
            await _hold_control(self, self.run, self.hers, key, time.monotonic(), trying=True)

    def _new_receipt(self, previous: tuple, key: str) -> tuple[float, float]:
        current = self._receipts()
        if not current or current != (previous + (current[-1],))[-128:]:
            raise _LostEvidenceError("missing_or_interfering_delivery")
        entry = current[-1]
        if (type(entry) is not tuple or len(entry) != 3 or entry[2] != key
                or any(type(t) not in (int, float) or not math.isfinite(t) for t in entry[:2])
                or not 0 <= entry[0] <= entry[1]):
            raise _LostEvidenceError("invalid_delivery_receipt")
        if previous and entry == previous[-1]:
            raise _LostEvidenceError("reused_delivery_receipt")
        return entry[0], entry[1]

    async def _release(self, ack_s: float, *, cleanup: bool = False) -> tuple[float, float] | None:
        owner = self._owned_press
        if owner is None:
            return None
        if self._pending_release is None:
            if (not cleanup and self.owns() is not True) or (self.run.held, self.run.held_since) != owner:
                raise _LostEvidenceError("input_owner_changed")
            self._release_previous = self._receipts()
            self._pending_release = asyncio.create_task(release_control_if_owned(
                self, self.run, self.hers, *owner, trying=True))
        released = await asyncio.wait_for(asyncio.shield(self._pending_release), ack_s)
        if not released:
            raise _LostEvidenceError("input_owner_changed")
        delivery = self._new_receipt(self._release_previous, '')
        self._pending_release = None
        self._owned_press = None
        self.release_problem = None
        return delivery

    async def _finish(self, ack_s: float) -> bool:
        """Join an in-flight delivery before releasing only its retained owner."""
        pending = self._pending
        if pending is not None:
            try:
                await asyncio.wait_for(asyncio.shield(pending), ack_s)
                delivery = self._new_receipt(self._pending_previous, self._pending_key)
                owner = self._pending_key, delivery[1]
                if delivery[0] < self._pending_began or (self.run.held, self.run.held_since) != owner:
                    raise _LostEvidenceError("input_owner_changed")
                self._owned_press = owner
            except Exception as why:  # noqa: BLE001 - the physical acknowledgement is unknown
                if not self._dispatch_attempted:
                    self._pending = None
                    return True
                self.release_problem = "dispatch_acknowledgement_unknown"
                logger.warning("pixel experiment delivery remains unconfirmed: %s", why)
                return False
            self._pending = None
            self._dispatch_attempted = False
        try:
            await self._release(ack_s, cleanup=True)
        except Exception as why:  # noqa: BLE001 - retain a failed physical release as an explicit result
            self.release_problem = ("release_acknowledgement_unknown" if self._pending_release is not None
                                    and not self._pending_release.done() else "release_owner_or_receipt_changed")
            logger.warning("pixel experiment release remains unconfirmed: %s", why)
            return False
        return True


async def _cleanup_collection(inputs: PixelExperimentInputs, ack_s: float, owner: object) -> bool:
    """Retire the collector only when its bounded cleanup has actually finished."""
    try:
        return await inputs._finish(ack_s)
    finally:
        if inputs._collector is owner:
            inputs._collector = None


async def _joined_cleanup(inputs: PixelExperimentInputs, ack_s: float) -> tuple[bool, bool]:
    cleanup = asyncio.create_task(_cleanup_collection(inputs, ack_s, inputs._collector))
    inputs._cleanup = cleanup
    cancelled = False
    while not cleanup.done():
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            cancelled = True
        except Exception:  # noqa: BLE001 - consume and report the finished cleanup's exception below
            break
    try:
        return cleanup.result(), cancelled
    except asyncio.CancelledError:
        inputs.release_problem = "cleanup_cancelled"
        return False, True
    except Exception as why:  # noqa: BLE001 - cleanup failure cannot certify a physical release
        inputs.release_problem = "cleanup_failed"
        logger.warning("pixel experiment cleanup failed: %s", why)
        return False, cancelled


def _guard(inputs: PixelExperimentInputs, interference: int, receipts: tuple,
           epoch: int, *, held: tuple[str, float] | None = None) -> None:
    if (inputs.owns() is not True or inputs._interference != interference
            or inputs._receipts() != receipts
            or inputs.hers.identification.epoch != epoch):
        raise _LostEvidenceError("input_interrupted")
    if held is not None:
        if (inputs.run.held, inputs.run.held_since) != held:
            raise _LostEvidenceError("input_owner_changed")
    elif not inputs._idle():
        raise _LostEvidenceError("neutral_owner_changed")


async def _captured(look: Callable[[], Awaitable[Any]], seconds: float, *, not_before: float) -> FrozenPixelFrame:
    deadline = time.monotonic() + seconds
    for _ in range(8):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("fresh capture budget ended")
        captured = await asyncio.wait_for(look(), remaining)
        if not isinstance(captured, tuple) or len(captured) != 2:
            raise _LostEvidenceError("missing_capture")
        frame = freeze_frame(*captured)
        if frame is None or frame.arrived > time.monotonic():
            raise _LostEvidenceError("unmeasured_capture")
        if frame.lower >= not_before:
            return frame
    raise _LostEvidenceError("queued_capture_budget")


async def _one_press(experiment: RandomizedPixelExperiment, inputs: PixelExperimentInputs,
                     look: Callable[[], Awaitable[Any]], neutral_s: float, settle_s: float,
                     capture_s: float, ack_s: float) -> None:
    scheduled = experiment.next_press
    neutral_began = time.monotonic()
    interference, receipts = inputs._interference, inputs._receipts()
    epoch = inputs.hers.identification.epoch
    _guard(inputs, interference, receipts, epoch)
    await asyncio.sleep(neutral_s)
    before = await _captured(look, capture_s, not_before=neutral_began)
    _guard(inputs, interference, receipts, epoch)
    if before.lower < neutral_began or time.monotonic() - before.upper > .75:
        raise _LostEvidenceError("before_outside_neutral")
    down_dispatch, down_ack = await inputs._deliver(scheduled.key, ack_s)
    if inputs._interference != interference or inputs.hers.identification.epoch != epoch:
        raise _LostEvidenceError("input_interrupted")
    receipts, held = inputs._receipts(), inputs._owned_press
    await asyncio.sleep(settle_s)
    after = await _captured(look, capture_s, not_before=down_ack + .02)
    _guard(inputs, interference, receipts, epoch, held=held)
    release = await inputs._release(ack_s)
    if inputs._interference != interference or inputs.hers.identification.epoch != epoch:
        raise _LostEvidenceError("input_interrupted")
    if release is None or not experiment.record(PixelPress(
            scheduled, down_dispatch, down_ack, *release, neutral_began, before, after)):
        raise _LostEvidenceError(experiment.refused or "missing_release")


def _failure_reason(why: Exception) -> str:
    if isinstance(why, _LostEvidenceError):
        return str(why)
    return "capture_or_delivery_timeout" if isinstance(why, TimeoutError) else "capture_or_delivery_failed"


async def collect_independent_pixel_trials(experiment: RandomizedPixelExperiment, *,
        look: Callable[[], Awaitable[Any]], inputs: PixelExperimentInputs,
        after_press: Callable[[PixelPress], Awaitable[bool]] | None = None,
        neutral_s: float = .04, settle_s: float = .04, capture_s: float = .3,
        ack_s: float = .3, total_s: float = 20.) -> PixelCollectionResult:
    """Use an already selected schedule, with finite capture and input waits.

    ``look`` can be ``lambda: canvas_frames.look(clip)``. The caller must hold
    its exclusive action lease and use ``inputs`` for all actions throughout
    this call. Cleanup has one further finite acknowledgement allowance.
    """
    if not isinstance(experiment, RandomizedPixelExperiment) or not isinstance(inputs, PixelExperimentInputs):
        raise TypeError("pixel collection requires an experiment and owned physical inputs")
    if (not callable(look) or after_press is not None and not callable(after_press)
            or any(type(v) not in (int, float) or not math.isfinite(v)
                                 for v in (neutral_s, settle_s, capture_s, ack_s, total_s))
            or not .02 <= neutral_s <= .2 or not .02 <= settle_s <= .2
            or not .01 <= capture_s <= .4 or not .01 <= ack_s <= .4 or not 0 < total_s <= 20.):
        raise ValueError("pixel collection needs finite observation and delivery budgets")
    if inputs._collector is not None:
        experiment.retire("collector_already_active")
        return PixelCollectionResult(experiment.refused, len(experiment.trials), False, inputs.release_problem)
    if len(experiment.schedule) > 24 or experiment.trials or experiment.refused or not inputs._idle():
        experiment.retire("experiment_not_fresh_or_neutral")
        return PixelCollectionResult(experiment.refused, len(experiment.trials), False, inputs.release_problem)
    inputs._collector = object()
    status = "complete"
    try:
        async with asyncio.timeout(total_s):
            while experiment.next_press is not None:
                await _one_press(experiment, inputs, look, neutral_s, settle_s, capture_s, ack_s)
                if after_press is not None:
                    interference, receipts = inputs._interference, inputs._receipts()
                    epoch = inputs.hers.identification.epoch
                    measured = await after_press(experiment.trials[-1])
                    _guard(inputs, interference, receipts, epoch)
                    if measured is not True:
                        raise _LostEvidenceError(experiment.refused or "pixel_measurement_unavailable")
    except asyncio.CancelledError:
        experiment.retire("cancelled")
        raise
    except Exception as why:  # noqa: BLE001 - any unavailable physical or capture receipt retires this experiment
        status = _failure_reason(why)
        experiment.retire(status)
    finally:
        released, cancelled = await _joined_cleanup(inputs, ack_s)
        if not released:
            experiment.retire("release_unconfirmed")
        if cancelled:
            experiment.retire("cancelled")
            raise asyncio.CancelledError
    return PixelCollectionResult(experiment.refused or status, len(experiment.trials), released, inputs.release_problem)
