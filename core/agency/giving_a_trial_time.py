"""Keep a control trial intact for the time it needs to answer.

Offline 2026-10-12 a turning trial was let go in the next picture: planning
resumed before any settled motion could show what the turn did. A trial asks
for one control for a measured time. A changed input or observation identity
ends it, and its time is bounded even when no answer comes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from core.agency.calibration_for_a_current_body import (
    BodyOwner,
    calibration_for_current_body,
    calibration_owner,
    uses_total_calibration_cap,
)
from core.agency.time_to_observe_a_control import observed_response_seconds
from core.agency.which_way_she_faces import LONGEST_TURN_S, SETTLE_S
from core.verify import invariant


@dataclass(frozen=True)
class ControlTrial:
    """One delivered control and the observation interval belonging to it."""

    key: str
    began: float
    until: float
    epoch: int
    owner: BodyOwner | None = None

    def holds(self, held: str, epoch: int, at: float) -> bool:
        """Whether this input still owns the current observation, within its finite time."""
        return (self.key == held and self.epoch == epoch
                and all(math.isfinite(v) for v in (self.began, self.until, at))
                and self.began <= at < self.until)

    def waits_for_picture(self, held: str, since: float, epoch: int, at: float) -> bool:
        """An old picture cannot interrupt the delivered input still waiting to be observed."""
        return self.belongs_to_input(held, since, epoch) and math.isfinite(at) and at < self.until

    def belongs_to_input(self, held: str, since: float, epoch: int) -> bool:
        """The delivered trial remains the same input while its phase is being completed."""
        return (self.key == held and self.epoch == epoch and self.began == since
                and all(math.isfinite(v) for v in (self.began, self.until, since)) and self.began < self.until)


async def try_a_facing_control(hands: Any, run: Any, hers: Any, choosing: Any, at: float, *,
                              hold: Any, seconds: float, limit: int) -> bool:
    """Hold the requested trial between sightings of where the body faces, then try the next one."""
    trial = run.facing_trial
    if run.faces is None or choosing.mine is None or hers.follows_pointer:
        run.facing_trial = None
        return False
    owner = calibration_owner(hers, choosing.mine)
    if owner is None and not uses_total_calibration_cap(hers, choosing.mine):
        run.facing_trial = None
        return False
    epoch = hers.identification.epoch
    programme = calibration_for_current_body(run, hers, choosing.mine, minimum=limit)
    if (trial is not None and trial.owner == owner and run.held_trying
            and trial.waits_for_picture(run.held, run.held_since, epoch, at)):
        return True
    run.facing_trial = None
    if owner is not None and programme is None:
        return False
    seconds = observed_response_seconds(getattr(run, "intervals", ()), least_s=seconds, settle_s=SETTLE_S,
                                        samples=4, span_s=.15, longest_s=max(4 * seconds, LONGEST_TURN_S)) or seconds
    if ((programme.issued >= programme.limit if programme is not None else run.facing_trials >= limit)
            or at - run.tried_at < seconds):
        return False
    key = run.faces.trial(run.keys)
    if key is None:
        return False
    if programme is not None and not programme.take_trial():
        return False
    run.facing_trials += 1
    if key and key == run.held:
        # A new experiment needs a new physical receipt; rechoosing the held
        # input otherwise reuses its old deadline and may end immediately.
        await hold(hands, run, hers, '', at, trying=False)
    await hold(hands, run, hers, key, at, trying=True)
    began = run.held_since if run.held_since is not None else at
    run.tried_at = began
    run.facing_trial = ControlTrial(key, began, began + seconds, epoch, owner)
    from core.agency.a_trial_has_a_physical_end import lease_control_trial

    lease_control_trial(hands, run, hers, run.facing_trial, field='facing_trial')
    run.longest_control_trial_s = max(getattr(run, "longest_control_trial_s", 0.), seconds)
    return True


@invariant("agency.control_trials_keep_their_measurement_time", scope="agency",
           owner="core/agency/giving_a_trial_time.py", observational=False)
def _control_trial_time_invariant() -> tuple:
    trial = ControlTrial("a", 1.0, 1.5, 3)
    assert trial.holds("a", 3, 1.25)
    assert not trial.holds("b", 3, 1.25)
    assert not trial.holds("a", 4, 1.25)
    assert not trial.holds("a", 3, 1.5)
    assert not ControlTrial("a", 1, math.inf, 3).holds("a", 3, 1.25)
    return ()
