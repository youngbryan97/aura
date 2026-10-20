"""Choose discovery or measured protection within one observation deadline.

Picture edges bound what can be measured, not where collision occurs.
This records predicted observation reserve; unmeasured dynamics remain
unknown. Existing physical leases still own cancellation and key release.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from core.agency.keeping_a_trial_observable import control_before_the_next_trial, visible_path_clearance
from core.perception.translation_from_visible_support import motion_was_measured
from core.verify import invariant


@dataclass(frozen=True)
class TrialDecision:
    kind: str
    reason: str
    at: float
    deadline: float
    owner: tuple | None
    input_owner: tuple | None
    wait_s: float
    key: str | None = None
    clearance: float | None = None


def acceleration_is_admitted(hers: Any, mine: Any, at: float) -> bool:
    """A selected law or its still-current predictive admission may protect observation."""
    if mine is None or mine.seen != at or not motion_was_measured(mine, at):
        return False
    if getattr(hers, 'carried', False):
        return True
    family = getattr(hers, 'motion_family_reading', None)
    models = getattr(getattr(getattr(hers, '_hers', None), 'intervals', None), 'families', None)
    prediction = getattr(family[2], 'prediction', None) if isinstance(family, tuple) and len(family) == 3 else None
    return (prediction is not None and prediction.selected is True and models is not None
            and models.selected is True and models.owner == prediction.owner
            and prediction.owner[:2] == (mine.number, getattr(mine, 'born', None))
            and family[1] == mine.number and family[0] <= at)


def decide_observation_reserve(run: Any, hers: Any, moves: Any, at: float, wait_s: float, *, active: bool) -> TrialDecision:
    """Use the same measured wait, body and admitted law for protection and discovery."""
    mine = hers.thing(moves)
    owner = (mine.number, getattr(mine, 'born', None), hers.identification.epoch) if mine is not None else None
    trial = getattr(run, 'body_trial', None)
    input_owner = (trial.key, trial.began, trial.until, trial.epoch) if active and trial is not None else None
    kind = 'complete_owned_trial' if active else 'bounded_experiment'
    if not math.isfinite(wait_s) or wait_s <= 0 or not math.isfinite(at):
        return TrialDecision('need_observation', 'unmeasured_deadline', at, at, owner, input_owner, wait_s)
    args = dict(at=float(at), deadline=float(at+wait_s), owner=owner, input_owner=input_owner, wait_s=float(wait_s))
    if not acceleration_is_admitted(hers, mine, at):
        return TrialDecision(kind, 'acceleration_reserve_unavailable', **args)
    rest = hers.push_of('')
    if rest is None or len(rest) != 2 or not all(math.isfinite(v) for v in rest):
        return TrialDecision(kind, 'neutral_response_unavailable', **args)
    values = (mine.x, mine.y, mine.vx, mine.vy, mine.w, mine.h, *moves.shape)
    if not all(math.isfinite(v) for v in values) or min(mine.w, mine.h, *moves.shape) <= 0:
        return TrialDecision('need_observation', 'current_geometry_unavailable', **args)
    protection = control_before_the_next_trial(hers, moves, at, run.keys, wait_s, measured_rest=rest)
    clearance = visible_path_clearance(mine, moves.shape, [(wait_s, rest)])
    if protection is not None:
        return TrialDecision('abort_for_protection' if active else 'protect_observation',
            'waiting_then_braking_loses_observation', key=protection.key, clearance=clearance, **args)
    if clearance <= 0:
        return TrialDecision('need_observation', 'no_verified_control_keeps_observation', clearance=clearance, **args)
    return TrialDecision(kind, 'predicted_neutral_observation_reserve', clearance=clearance, **args)


@invariant('agency.discovery_records_its_observation_deadline_and_owner', scope='agency',
           owner='core/agency/a_control_trial_and_its_observation_reserve.py', observational=False)
def _trial_reserve_invariant() -> tuple:
    from types import SimpleNamespace

    body = SimpleNamespace(number=1, born=0., seen=1., x=120., y=170., vx=0., vy=100., w=14., h=14.)
    pushes = {'': (0., 90.), 'q': (0., -110.)}
    hers = SimpleNamespace(carried=True, thing=lambda _moves: body, push_of=pushes.get,
                           identification=SimpleNamespace(epoch=2))
    run, moves = SimpleNamespace(keys=['q'], body_trial=None), SimpleNamespace(shape=(240, 240))
    decision = decide_observation_reserve(run, hers, moves, 1., .65, active=False)
    assert decision.kind == 'protect_observation' and decision.key == 'q'
    assert decision.owner == (1, 0., 2) and decision.deadline == 1.65
    pushes.pop('q')
    assert decide_observation_reserve(run, hers, moves, 1., .65, active=False).kind == 'need_observation'
    return ()
