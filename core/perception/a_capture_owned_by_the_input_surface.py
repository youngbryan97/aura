"""Carry input-surface qualification through fresh captures of changing scenes.

The adapter owns capture and delivery for the surface. An ordinary positive
place check qualifies that surface within its active handover. A later
capture can retain that qualification without claiming the same scene,
body, button, task outcome or accepted input.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from core.perception.observed_transfer import ScreenReading, ScreenSnapshot, freeze_screen
from core.verify import invariant

MAX_CAPTURE_AGE_S = 10.


@dataclass(frozen=True)
class _OwnedCapture:
    issuer: InputSurfaceCaptures = field(repr=False, compare=False)
    observation: ScreenReading = field(repr=False, compare=False)
    frame: ScreenSnapshot = field(repr=False)
    identity: str


class InputSurfaceCaptures:
    """One active adapter; at most one current capture and one qualified context."""

    def __init__(self, identity: Callable[[], str | None]) -> None:
        self.identity = identity
        self.current: _OwnedCapture | None = None
        self.qualified: tuple | None = None
        self.permitted: _OwnedCapture | None = None

    def invalidate(self, *, retire: bool = False) -> None:
        self.current = None
        self.permitted = None
        if retire:
            self.qualified = None
            self.permitted = None

    def delivered(self, succeeded: bool, *, now: float | None = None) -> None:
        """A port acknowledgement retains surface permission, separately from physical effect."""
        permitted, self.permitted = self.permitted, None
        if (succeeded is True and permitted is not None and self.identity() == permitted.identity
                and _current_pixels(permitted.observation, permitted.frame, time.time() if now is None else now)):
            self.qualified = permitted.frame.surface, permitted.frame.work_epoch
        elif succeeded is not True:
            self.qualified = None

    def bind(self, observation: Any, *, now: float | None = None) -> bool:
        """Only the capture adapter calls this after receiving its measured reading."""
        self.invalidate()
        if not isinstance(observation, ScreenReading):
            return False
        frame = freeze_screen(observation)
        identity = self.identity()
        if (not identity or dict(frame.surface).get('surface_id') != identity
                or not _current_pixels(observation, frame, time.time() if now is None else now)):
            return False
        receipt = _OwnedCapture(self, observation, frame, identity)
        self.current = receipt
        observation._input_surface_capture = receipt
        return True

    def recognizes(self, receipt: _OwnedCapture, observation: Any, *, now: float) -> bool:
        """Verify active ownership and the exact retained capture, rather than public flags."""
        if receipt is not self.current or receipt.observation is not observation or self.identity() != receipt.identity:
            return False
        frame = freeze_screen(observation)
        return (_current_pixels(observation, frame, now)
                and frame.picture is receipt.frame.picture
                and (frame.surface, frame.bounds, frame.capture_at, frame.epoch, frame.work_epoch)
                    == (receipt.frame.surface, receipt.frame.bounds, receipt.frame.capture_at,
                        receipt.frame.epoch, receipt.frame.work_epoch))


def _current_pixels(observation: Any, frame: ScreenSnapshot, now: float) -> bool:
    return (observation.get('ok') is True and frame.picture is not None and frame.bounds is not None
            and bool(frame.surface) and bool(observation.get('_capture_epoch')) and not frame.occluded
            and observation.get('her_window_showing') is not False and not observation.get('refused_because')
            and frame.capture_at is not None and math.isfinite(now)
            and 0 <= now - frame.capture_at <= MAX_CAPTURE_AGE_S)


def _receipt(observation: Any) -> _OwnedCapture | None:
    receipt = getattr(observation, '_input_surface_capture', None)
    return receipt if isinstance(receipt, _OwnedCapture) else None


def qualify_current_input_surface(observation: Any, *, now: float | None = None) -> bool:
    """Called after the ordinary place check was positive; never from a public field."""
    receipt = _receipt(observation)
    if receipt is None or not receipt.issuer.recognizes(receipt, observation, now=time.time() if now is None else now):
        return False
    receipt.issuer.permitted = receipt
    return True


def current_input_surface_is_qualified(observation: Any, *, now: float | None = None) -> bool:
    """Environment admission only; changing scenes need their own task interpretation."""
    receipt = _receipt(observation)
    return (receipt is not None and receipt.issuer.qualified == (receipt.frame.surface, receipt.frame.work_epoch)
            and receipt.issuer.recognizes(receipt, observation, now=time.time() if now is None else now))


@invariant('perception.input_surface_authority_needs_its_current_owned_capture', scope='perception',
           owner='core/perception/a_capture_owned_by_the_input_surface.py', observational=False)
def _owned_input_surface_invariant() -> tuple:
    from core.perception.observed_transfer import PixelSnapshot

    def observed(at):
        return ScreenReading(dict(ok=True, surface_id='one', bounds=[0, 0, 10, 10], capture_at=at,
                                  _capture_epoch=str(at)), picture_snapshot=PixelSnapshot(bytes(768), 16, 16))
    owner = InputSurfaceCaptures(lambda: 'one')
    first, next_view = observed(1.), observed(2.)
    assert owner.bind(first, now=1.1) and not current_input_surface_is_qualified(first, now=1.1)
    assert qualify_current_input_surface(first, now=1.1)
    assert not current_input_surface_is_qualified(first, now=1.1)
    owner.delivered(True, now=1.1)
    assert owner.bind(next_view, now=2.1) and current_input_surface_is_qualified(next_view, now=2.1)
    assert not current_input_surface_is_qualified(first, now=2.1)
    assert not current_input_surface_is_qualified(dict(next_view), now=2.1)
    owner.invalidate()
    assert not current_input_surface_is_qualified(next_view, now=2.1)
    return ()
