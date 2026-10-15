"""Share finite control trials across the current body's unknown inputs."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.agency.calibration_for_a_current_body import (
    MAX_CALIBRATION_CONTROLS,
    MAX_CALIBRATION_OWNERS,
    BodyOwner,
    calibration_owner,
)
from core.agency.controls_ready_for_the_motion_model import has_a_measured_response
from core.verify import invariant

PRESSES_PER_CONTROL = 2


def _bounded_controls(values: Any) -> tuple[str, ...] | None:
    if isinstance(values, (str, bytes)):
        return None
    try:
        values = iter(values)
    except TypeError:
        return None
    found = {}
    for index, key in enumerate(values):
        if index >= 4 * MAX_CALIBRATION_CONTROLS:
            return None
        if not isinstance(key, str) or not key or len(key) > 64:
            return None
        found.setdefault(key, None)
        if len(found) > MAX_CALIBRATION_CONTROLS:
            return None
    return tuple(found)


@dataclass
class UnknownControlTrials:
    """An issued request consumes its allowance even without a usable outcome."""

    owner: BodyOwner
    issued: dict[str, int] = field(default_factory=dict)

    def available(self, unknown: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(self.issued) | set(unknown)) > MAX_CALIBRATION_CONTROLS:
            return ()
        return tuple(key for key in unknown if self.issued.get(key, 0) < PRESSES_PER_CONTROL)

    def take(self, unknown: tuple[str, ...]) -> str | None:
        eligible = self.available(unknown)
        if not eligible:
            return None
        key = min(eligible, key=lambda control: self.issued.get(control, 0))
        self.issued[key] = self.issued.get(key, 0) + 1
        return key


def next_unknown_control(run: Any, hers: Any, mine: Any, unknown: Any) -> str | None:
    """Reserve one current authorized input without renewing an old body's budget."""
    owner = calibration_owner(hers, mine)
    authorized = _bounded_controls(run.keys)
    candidates = _bounded_controls(unknown)
    programmes = getattr(run, '_unknown_control_trials', None)
    if programmes is None:
        programmes = run._unknown_control_trials = {}
    if owner is None or authorized is None or candidates is None:
        run._unknown_control_probe_status = 'current body or control receipt unavailable'
        return None
    candidates = tuple(key for key in candidates if key in authorized)
    if not candidates:
        run._unknown_control_probe_status = 'no authorized unknown control'
        return None
    if owner not in programmes:
        if len(programmes) >= MAX_CALIBRATION_OWNERS:
            run._unknown_control_probe_status = 'body owner budget exhausted'
            return None
        programmes[owner] = UnknownControlTrials(owner)
    key = programmes[owner].take(candidates)
    run._unknown_control_probe_status = 'trial reserved' if key is not None else 'unknown control budget exhausted'
    return key


def unknown_control_trials_finished(run: Any, hers: Any) -> bool:
    """Exhausted or unowned experiments yield after their current input and rest."""
    programmes = getattr(run, '_unknown_control_trials', None)
    if programmes is None:
        return False
    counter = getattr(run, 'counter_trial', None)
    if getattr(counter, 'key', '') and not getattr(counter, 'done', True):
        return False
    authorized = _bounded_controls(run.keys)
    mine = getattr(run, 'things', {}).get(getattr(hers, 'number', None))
    owner = calibration_owner(hers, mine)
    if owner is None or authorized is None:
        return True
    programme = programmes.get(owner)
    if programme is None:
        return len(programmes) >= MAX_CALIBRATION_OWNERS
    unknown = tuple(key for key in authorized if not has_a_measured_response(hers, key))
    return not programme.available(unknown)


@invariant('agency.unknown_controls_share_a_finite_current_body_budget', scope='agency',
           owner='core/agency/unknown_controls_get_their_turn.py', observational=False)
def _unknown_control_fairness_invariant() -> tuple:
    from types import SimpleNamespace

    run = SimpleNamespace(keys=['first', 'second', 'third'])
    hers = SimpleNamespace(number=4, identification=SimpleNamespace(epoch=2))
    mine = SimpleNamespace(number=4, born=1.)
    assert next_unknown_control(run, hers, mine, run.keys) == 'first'
    run.keys.reverse()
    assert next_unknown_control(run, hers, mine, run.keys) == 'third'
    assert next_unknown_control(run, hers, mine, run.keys) == 'second'
    for _ in range(3):
        assert next_unknown_control(run, hers, mine, run.keys) is not None
    assert next_unknown_control(run, hers, mine, run.keys) is None
    assert set(run._unknown_control_trials[(2, 4, 1.)].issued.values()) == {PRESSES_PER_CONTROL}
    run.keys.reverse()
    assert next_unknown_control(run, hers, mine, run.keys) is None
    from core.agency.inputs_while_measuring_control import body_probe_is_active

    run.things, run.trying = {4: mine}, 0
    hers.kind, hers.carried = 1, False
    hers.keys_known, hers.way_of = lambda _keys: False, lambda _key: None
    assert unknown_control_trials_finished(run, hers)
    assert not body_probe_is_active(run, hers, pointer_steps=0, at=2.)
    return ()
