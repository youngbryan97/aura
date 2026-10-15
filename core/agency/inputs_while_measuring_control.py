"""Keep isolated input experiments separate from actions performed together.

Offline 2026-10-12 attacks were delivered during movement trials. Changing
poses then answered a key that was never tried alone. A second input ends
the first input's measured interval; it cannot extend its causal receipts.
"""
from __future__ import annotations

import asyncio
import logging
import math
import time
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from typing import Any

from core.runtime.lockdep import checked_async_lock
from core.verify import invariant

logger = logging.getLogger(__name__)
_PRIMARY: ContextVar[frozenset[tuple[int, int]]] = ContextVar('primary_control_delivery', default=frozenset())
_INPUTS = frozenset({'down', 'up', 'tap', 'press', 'click', 'point'})


def body_probe_is_active(run: Any, hers: Any, *, pointer_steps: int, at: float | None = None) -> bool:
    """Reserve a finite discovery pass for one input at a time, including its rests."""
    if getattr(run, 'pointer_first', False):
        return not hers.follows_pointer and 0 <= getattr(run, 'pointed', pointer_steps) < pointer_steps
    trial = getattr(run, 'body_trial', None)
    epoch = getattr(getattr(hers, 'identification', None), 'epoch', 0)
    if (at is not None and trial is not None and getattr(run, 'held_trying', False)
            and trial.belongs_to_input(getattr(run, 'held', ''), getattr(run, 'held_since', None), epoch)
            and math.isfinite(at) and (at < trial.until or bool(trial.key))):
        return True
    keys = getattr(run, 'keys', ())
    trying = getattr(run, 'trying', None)
    if getattr(hers, 'number', None) is not None:
        from core.agency.calibration_for_a_current_body import calibration_owner
        from core.agency.unknown_controls_get_their_turn import unknown_control_trials_finished

        if unknown_control_trials_finished(run, hers):
            return False
        mine = getattr(run, 'things', {}).get(hers.number)
        if calibration_owner(hers, mine) is not None:
            # Discovery before this owner was found cannot spend its finite
            # allowance for measuring the controls it has not learned yet.
            return bool(keys and type(trying) is int and trying >= 0 and not hers.keys_known(keys))
    return bool(keys and type(trying) is int and 0 <= trying < 4 * len(keys)
                and (hers.number is None or not hers.keys_known(keys)))


@invariant("agency.control_discovery_keeps_its_finite_observation_phase", scope="agency",
           owner="core/agency/inputs_while_measuring_control.py", observational=False)
def _body_probe_phase_invariant() -> tuple:
    from types import SimpleNamespace

    from core.agency.giving_a_trial_time import ControlTrial
    from core.agency.playing_as_it_happens import _still_trying_keys

    hers = SimpleNamespace(number=None, kind=None, follows_pointer=False,
                           identification=SimpleNamespace(epoch=0), keys_known=lambda _keys: False)
    run = SimpleNamespace(keys=['a', 'b'], trying=4, pointer_first=False, body_trial=None)
    choosing = SimpleNamespace(mine=None)
    assert _still_trying_keys(run, hers, choosing, False, at=1.)
    run.trying, run.held, run.held_since, run.held_trying = 8, '', 1., True
    run.body_trial = ControlTrial('', 1., 1.5, 0)
    assert _still_trying_keys(run, hers, choosing, False, at=1.25)
    assert not _still_trying_keys(run, hers, choosing, False, at=1.5)
    return ()


@invariant("agency.an_observed_body_trial_finishes_with_its_rest", scope="agency",
           owner="core/agency/inputs_while_measuring_control.py", observational=False)
def _body_trial_rest_invariant() -> tuple:
    from types import SimpleNamespace

    from core.agency.giving_a_trial_time import ControlTrial

    hers = SimpleNamespace(number=1, follows_pointer=False, identification=SimpleNamespace(epoch=0),
                           keys_known=lambda _keys: True)
    run = SimpleNamespace(keys=['a'], trying=4, held='a', held_since=1., held_trying=True,
                          body_trial=ControlTrial('a', 1., 1.5, 0))
    assert body_probe_is_active(run, hers, pointer_steps=6, at=1.6)
    run.held, run.held_since, run.body_trial = '', 1.6, ControlTrial('', 1.6, 2., 0)
    assert body_probe_is_active(run, hers, pointer_steps=6, at=1.8)
    assert not body_probe_is_active(run, hers, pointer_steps=6, at=2.)
    return ()


class HandsWithControlReceipts:
    """Observe delivered input boundaries without changing what the hands do."""

    def __init__(self, hands: Any, run: Any, hers: Any) -> None:
        self.hands, self.run, self.hers = hands, run, hers

    def __getattr__(self, name: str) -> Any:
        original = getattr(self.hands, name)
        if name not in _INPUTS or not callable(original):
            return original

        async def delivered(*args: Any, **kwargs: Any) -> Any:
            if (id(self), id(asyncio.current_task())) not in _PRIMARY.get():
                await _end_prepared_motion(self, self.run, self.hers)
                await _end_reflex_response(self, self.run, self.hers)
                self._other_input(name, time.monotonic())
            return await original(*args, **kwargs)

        return delivered

    def _other_input(self, name: str, at: float) -> None:
        if hasattr(self.run, 'previous_control'):
            self.run.previous_control = None
        held = self.hers._held_at(at)
        if held is not None:
            began, key, _trying = held
            for record in (*self.hers._by_thing.values(), self.hers._hers):
                record.end(key, began)
        self.run.presses.interrupted()
        if name != 'point':
            self.hers._followed.clear()
            self.hers._sighted = None
        logger.debug("single-input observation ended before %s", name)


@contextmanager
def _primary_delivery(hands: HandsWithControlReceipts):
    hands.hers._followed.clear()
    hands.hers._sighted = None
    token = _PRIMARY.set(_PRIMARY.get() | {(id(hands), id(asyncio.current_task()))})
    try:
        yield
    finally:
        _PRIMARY.reset(token)


def primary_delivery(hands: Any):
    """Only the trial's own input delivery may continue its independent interval."""
    return _primary_delivery(hands) if isinstance(hands, HandsWithControlReceipts) else nullcontext()


async def hold_control(hands: Any, run: Any, hers: Any, key: str, at: float, *, trying: bool = False,
                       now: Any = None) -> None:
    await _end_prepared_motion(hands, run, hers)
    await _end_reflex_response(hands, run, hers)
    if key == run.held:
        return
    async with _control_delivery_lock(run):
        await _hold_control(hands, run, hers, key, at, trying=trying, now=now)


async def _end_prepared_motion(hands: Any, run: Any, hers: Any) -> None:
    if getattr(run, '_prepared_input_set', None) is None:
        return
    from core.agency.acting_on_a_prepared_motion import abort_prepared_motion

    if not await abort_prepared_motion(hands, run, hers):
        raise RuntimeError("a prepared input's physical release remains unconfirmed")


async def _end_reflex_response(hands: Any, run: Any, hers: Any) -> None:
    if getattr(run, '_reflex_input', None) is None:
        return
    from core.agency.responding_before_slower_planning import abort_reflex_response

    await abort_reflex_response(hands, run, hers)


def _control_delivery_lock(run: Any) -> Any:
    lock = getattr(run, '_control_delivery_lock', None)
    if lock is None:
        lock = run._control_delivery_lock = checked_async_lock('realtime_control_delivery')
    return lock


async def release_control_if_owned(hands: Any, run: Any, hers: Any, key: str, since: float, *,
                                   trying: bool = False, trial_epoch: int | None = None) -> bool:
    """A physical input deadline can release only its own delivered receipt."""
    async with _control_delivery_lock(run):
        if run.held != key or run.held_since != since:
            return False
        await _hold_control(hands, run, hers, '', time.monotonic(), trying=trying, trial_epoch=trial_epoch)
        return True


async def _hold_control(hands: Any, run: Any, hers: Any, key: str, at: float, *, trying: bool = False,
                        now: Any = None, trial_epoch: int | None = None) -> None:
    """A changed physical input starts one trial; a changed policy cannot."""
    if key == run.held:
        return
    from core.agency.a_trial_has_a_physical_end import cancel_trial_deadline

    cancel_trial_deadline(run)
    now = now or time.monotonic
    began = now()
    with primary_delivery(hands):
        if run.held:
            await hands.up(run.held)
            released_at = now()
            _input_clock_receipt(run, began, released_at, '')
            _end_modelled_observation(run, released_at)
            if run.presses is not None:
                run.presses.released(run.held, released_at)
        dispatched = now()
        if not run.held:
            _end_modelled_observation(run, dispatched)
        if key:
            await hands.down(key)
            run.input_key_downs[key] += 1
        if trial_epoch is not None and hers.identification.epoch != trial_epoch:
            trying = False
        if run.presses is not None:
            run.presses.pressed(key, now(), run.things.values(), hers.number,
                                epoch=hers.identification.epoch, independent=trying)
    delivered = now()
    if key:
        _input_clock_receipt(run, dispatched, delivered, key)
    run.responses.append(max(0.0, delivered - at))
    timing = getattr(run, 'control_timing', None)
    if timing is not None:
        run.control_delivery = timing.delivered(at, began, delivered)
    if key:
        hers.tapped(key, delivered, began=dispatched)
    run.held, run.held_since, run.held_trying = key, delivered, trying
    hers.holding(key, delivered, trying=trying)


def _input_clock_receipt(run: Any, began: float, ended: float, key: str) -> None:
    receipts = getattr(run, '_input_clock_receipts', [])
    receipts.append((began, ended, key))
    run._input_clock_receipts = receipts[-128:]


def _end_modelled_observation(run: Any, at: float) -> None:
    """Close a forecast at release acknowledgement or the end of its neutral interval."""
    active = getattr(run, '_modelled_steering', None)
    if active is None:
        return
    action, _until, since, model = active
    if run.held == action.key and run.held_since == since:
        model.ended(action, since, at)


async def release_after_play(hands: Any, run: Any, hers: Any) -> None:
    """Stop observation tasks and close every input when its play owner leaves."""
    from core.agency.a_trial_has_a_physical_end import cancel_trial_deadline

    await _end_prepared_motion(hands, run, hers)
    cancel_trial_deadline(run)
    await _end_reflex_response(hands, run, hers)
    for field in ('_steering_input_deadline', '_reflex_input_deadline', 'reading'):
        task = getattr(run, field, None)
        if task is not None:
            task.cancel()
    held = [run.held] if run.held else []
    for key in [*held, *run.letting_go, *([run.charging.key] if run.charging.key else [])]:
        try:
            if key == run.held:
                await hold_control(hands, run, hers, '', time.monotonic())
            else:
                await hands.up(key)
        except (RuntimeError, OSError, ValueError, TypeError, AttributeError) as why:
            logger.debug("letting go of %s failed: %s", key, why)


@invariant("agency.a_second_input_ends_an_isolated_body_trial", scope="agency",
           owner="core/agency/inputs_while_measuring_control.py", observational=False)
def _single_input_interval_invariant() -> tuple:
    from core.agency.which_one_answers_to_her import _Speeds

    record = _Speeds()
    for at in (1.1, 1.2):
        record.add('a', 30., 0., press=1., when=at)
    before = list(record.by_press[('a', 1.)])
    record.end('a', 1.)
    record.add('a', -80., 0., press=1., when=1.3)
    record.unseen('a', 1., contradicted=True)
    assert record.by_press[('a', 1.)] == before
    assert record.timed[('a', 1.)][-1][0] == 1.2
    return ()
