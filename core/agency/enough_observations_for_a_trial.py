"""A finite owned-body experiment can finish when its measurements arrive."""
from __future__ import annotations

import math
from typing import Any

from core.agency.calibration_for_a_current_body import calibration_owner
from core.agency.which_one_answers_to_her import PUSH_OVER_S, PUSH_PICTURES, SETTLE_S
from core.verify import invariant


def trial_has_its_observations(trial: Any, hers: Any, moves: Any, at: float) -> bool:
    """Only the current owner's distinct settled samples can end its acquisition phase."""
    if (trial.expected_until is None or trial.independent or trial.owner is None
            or not math.isfinite(at)
            or not trial.began <= trial.expected_until <= trial.until):
        return False
    mine = hers.thing(moves)
    if trial.owner != calibration_owner(hers, mine):
        return False
    press = trial.key, trial.began
    if press in hers._hers.interrupted:
        return False
    samples = hers._hers.timed.get(press, ())
    if len(samples) < PUSH_PICTURES or any(not all(math.isfinite(v) for v in sample) for sample in samples):
        return False
    times = [sample[0] for sample in samples]
    return (trial.began + SETTLE_S <= times[0] and times[-1] <= at
            and all(b > a for a, b in zip(times, times[1:], strict=False))
            and times[-1] - times[0] >= PUSH_OVER_S)


def expected_trial_wait(trial: Any, at: float, next_picture_s: float) -> float:
    """Project the expected remaining acquisition time inside its unchanged hard end."""
    expected = trial.expected_until if trial.expected_until is not None else trial.until
    return min(max(0., trial.until - at), max(next_picture_s, expected - at))


@invariant('agency.trial_acquisition_requires_owned_samples', scope='agency',
           owner='core/agency/enough_observations_for_a_trial.py', observational=False)
def _trial_observation_goal_invariant() -> tuple:
    from types import SimpleNamespace

    from core.agency.giving_a_trial_time import ControlTrial
    from core.agency.which_one_answers_to_her import WhichIsHers

    hers = WhichIsHers()
    hers.number, hers.kind = 1, 1
    mine = SimpleNamespace(number=1, born=0.)
    moves = SimpleNamespace(things={1: mine})
    trial = ControlTrial('q', 1., 2.8, 0, (0, 1, 0.), False, 1.8)
    for at in (1.45, 1.58, 1.70):
        hers._hers.add('q', 0., at * 10., press=1., when=at)
    assert not trial_has_its_observations(trial, hers, moves, 1.9)
    hers._hers.add('q', 0., 20.5, press=1., when=2.05)
    assert trial_has_its_observations(trial, hers, moves, 2.05)
    assert trial.until == 2.8
    hers.identification.epoch += 1
    assert not trial_has_its_observations(trial, hers, moves, 2.05)
    return ()
