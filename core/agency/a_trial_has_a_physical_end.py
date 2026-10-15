"""End a measured input at its deadline even when pictures stop arriving.

Offline 2026-10-13 a discovery press crossed an interface transition and
stayed down for twelve seconds. Its observation programme was finite, but
only another picture could let the key go. A physical lease owns the exact
delivered press; expiry releases it and starts its planned observation rest.
"""
from __future__ import annotations

import asyncio
import math
import time
from typing import Any

from core.agency.inputs_while_measuring_control import release_control_if_owned
from core.runtime.task_ownership import fire_and_forget
from core.verify import invariant


def cancel_trial_deadline(run: Any) -> None:
    """Changing input retires the old lease, without cancelling its own expiry."""
    task = getattr(run, '_trial_input_deadline', None)
    run._trial_input_deadline = None
    run._trial_input_lease = None
    if task is not None and not task.done() and task is not asyncio.current_task():
        task.cancel()


def lease_control_trial(hands: Any, run: Any, hers: Any, trial: Any, *, field: str) -> bool:
    """A current delivered trial acquires a deadline separately from its sampling policy."""
    if field not in ('body_trial', 'facing_trial') or not trial.key:
        return False
    epoch = getattr(getattr(hers, 'identification', None), 'epoch', None)
    if (getattr(run, field, None) is not trial or getattr(run, 'held_trying', False) != trial.independent
            or not trial.belongs_to_input(run.held, run.held_since, epoch)):
        return False
    previous = getattr(run, '_trial_input_lease', None)
    if previous is not None and previous[:2] == (trial.key, trial.began):
        return False
    cancel_trial_deadline(run)
    run._trial_input_lease = trial.key, trial.began, trial.until
    run._trial_input_deadline = fire_and_forget(
        _end_trial(hands, run, hers, trial, field), name='measured_control_input_deadline',
        owner='realtime_control', bounded=True)
    return True


async def _end_trial(hands: Any, run: Any, hers: Any, trial: Any, field: str) -> None:
    """The lease performs one release; it never issues a subsequent press."""
    await asyncio.sleep(max(0., trial.until - time.monotonic()))
    if not _lease_matches(run, trial):
        return
    # The physical release remains necessary after an identity change. That
    # change prevents the old experiment from supplying a new measured rest.
    epoch = getattr(getattr(hers, 'identification', None), 'epoch', None)
    current = epoch == trial.epoch
    if not await release_control_if_owned(hands, run, hers, trial.key, trial.began,
                                         trying=current and trial.independent, trial_epoch=trial.epoch):
        return
    current = getattr(getattr(hers, 'identification', None), 'epoch', None) == trial.epoch
    if not current or field != 'body_trial' or getattr(run, field, None) is not trial:
        if getattr(run, field, None) is trial:
            setattr(run, field, None)
        return
    from core.agency.giving_a_trial_time import ControlTrial
    from core.agency.the_key_against_its_going import FIRST

    counter = getattr(run, 'counter_trial', None)
    if counter is not None:
        counter.released_at_deadline(trial.key, trial.began, run.held_since, epoch, steps=FIRST)
    # A release is a real neutral receipt. Keep its finite observation phase;
    # the next picture may finish it, but cannot skip it to press another key.
    expected = run.held_since + trial.expected_until - trial.began if trial.expected_until is not None else None
    setattr(run, field, ControlTrial('', run.held_since,
            run.held_since + trial.until - trial.began, epoch, trial.owner, trial.independent, expected))


def _lease_matches(run: Any, trial: Any) -> bool:
    return (bool(trial.key) and run.held_trying == trial.independent and run.held == trial.key
            and run.held_since == trial.began and math.isfinite(trial.until)
            and math.isfinite(trial.began) and trial.until > trial.began)


@invariant('agency.a_control_trial_has_a_physical_deadline', scope='agency',
           owner='core/agency/a_trial_has_a_physical_end.py', observational=False)
def _physical_trial_deadline_invariant() -> tuple:
    from types import SimpleNamespace

    from core.agency.giving_a_trial_time import ControlTrial

    trial = ControlTrial('opaque', 1., 1.5, 2)
    run = SimpleNamespace(held='opaque', held_since=1., held_trying=True)
    assert _lease_matches(run, trial)
    run.held_since = 1.1
    assert not _lease_matches(run, trial)
    run.held_since, run.held = 1., 'other'
    assert not _lease_matches(run, trial)
    run.held, run.held_trying = 'opaque', False
    assert not _lease_matches(run, trial)
    conditional = ControlTrial('opaque', 1., 1.5, 2, independent=False)
    assert _lease_matches(run, conditional)
    return ()
