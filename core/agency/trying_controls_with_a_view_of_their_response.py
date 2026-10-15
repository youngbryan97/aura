"""Deliver finite control experiments while retaining time to observe them."""
from __future__ import annotations

import statistics
import time
from typing import Any

from core.agency.a_response_worth_measuring_again import response_worth_repeating
from core.agency.a_trial_has_a_physical_end import lease_control_trial
from core.agency.calibration_for_a_current_body import calibration_owner
from core.agency.controls_ready_for_the_motion_model import has_a_measured_response
from core.agency.countering_motion_as_it_appears import CounteringMotion
from core.agency.enough_observations_for_a_trial import (
    expected_trial_wait,
    trial_has_its_observations,
)
from core.agency.giving_a_trial_time import ControlTrial
from core.agency.keeping_a_trial_observable import control_before_the_next_trial
from core.agency.the_key_against_its_going import FIRST, against_its_going
from core.agency.time_to_observe_a_control import observed_response_seconds, reserved_phase_seconds
from core.agency.trials_with_room_to_observe import trials_with_room
from core.agency.unknown_controls_get_their_turn import next_unknown_control
from core.agency.which_one_answers_to_her import (
    PUSH_OVER_S,
    PUSH_PICTURES,
    PUSHED_FOR_S,
    REALLY_MOVES,
    RESPONSE_S,
    SETTLE_S,
)
from core.perception.what_moves_in_the_picture import TOLD_FROM_STANDING_PX


async def try_controls(hands: Any, run: Any, hers: Any, at: float, moves: Any, *, hold: Any, seconds: float) -> None:
    """Repeat unknown controls when there is headroom, using measured control when waiting would lose it."""
    epoch = getattr(getattr(hers, "identification", None), "epoch", 0)
    maximum_s = 4 * seconds
    trial = getattr(run, "body_trial", None)
    active = (trial is not None and run.held_trying == trial.independent
              and trial.waits_for_picture(run.held, run.held_since, epoch, at))
    seconds = observed_response_seconds(getattr(run, "intervals", ()), least_s=seconds, settle_s=SETTLE_S,
                                        samples=PUSH_PICTURES, span_s=PUSH_OVER_S, longest_s=maximum_s) or seconds
    identified = hers is not None and moves is not None and hers.kind is not None and hers.thing(moves) is not None
    qualified = active and trial_has_its_observations(trial, hers, moves, at)
    if identified:
        run.things = moves.things
    if identified and hers.carried and not hers.keys_known(run.keys):
        cadence = statistics.median(run.intervals) if run.intervals else seconds / PUSH_PICTURES
        remaining = expected_trial_wait(trial, at, cadence) if active else seconds
        measured_delays = run.motion_responses if len(run.motion_responses) >= 3 else run.responses
        timing = getattr(run, "control_timing", None)
        current_delay = timing.ahead(at, time.monotonic()) if timing is not None else None
        delay = max(RESPONSE_S, current_delay if current_delay is not None else
                    statistics.median(measured_delays) if measured_delays else RESPONSE_S)
        observation = control_before_the_next_trial(hers, moves, at, run.keys, remaining + SETTLE_S + delay)
        if observation is not None:
            run.body_trial = None
            await hold(hands, run, hers, observation.key, at, trying=False)
            run.tried_at = run.held_since
            return
    if active:
        if not qualified:
            return
        if trial.key:
            await hold(hands, run, hers, '', at, trying=False)
            began = run.held_since
            expected_s = trial.expected_until - trial.began
            run.body_trial = ControlTrial('', began, began + trial.until - trial.began,
                epoch, trial.owner, False, began + expected_s)
            run.tried_at = began
            return
    run.body_trial = None
    if trial is None and at - run.tried_at < seconds:
        return
    if not hasattr(run, "counter_trial"):
        run.counter_trial = CounteringMotion()
    key = run.counter_trial.choose(run.keys, getattr(moves, "things", {}).values(), at,
                                   held=getattr(run, "held", ""), held_since=run.held_since,
                                   epoch=epoch,
                                   settle_s=SETTLE_S, step_s=seconds, steps=FIRST,
                                   wait_s=4 * len(run.keys) * seconds,
                                   against=lambda keys, things: against_its_going(keys, things, 2.0 * TOLD_FROM_STANDING_PX),
                                   measured=(lambda keys, things: response_worth_repeating(keys, things, hers, moves.shape,
                                                 at, run.held_since, seconds=seconds, steps=FIRST,
                                                 effect_over=REALLY_MOVES / PUSHED_FOR_S))
                                             if hers is not None and hasattr(moves, "shape") else None)
    if key is None:
        if identified:
            if trial is not None and run.held and trial.belongs_to_input(run.held, run.held_since, epoch):
                key = ""
            else:
                unknown = [control for control in run.keys if not has_a_measured_response(hers, control)]
                unknown = trials_with_room(unknown, hers, moves, at, seconds)
                key = '' if run.held and run.held in unknown else next_unknown_control(run, hers, hers.thing(moves), unknown)
                if key is None:
                    return
            run.trying += 1
        elif run.keys:
            turn = run.trying % (2 * len(run.keys))
            run.trying += 1
            key = run.keys[turn // 2] if turn % 2 == 0 else ""
        else:
            return
    counter_active = run.counter_trial.key and not run.counter_trial.done
    duration = reserved_phase_seconds(seconds, run.counter_trial.phase_seconds) if counter_active else seconds
    # State-dependent calibration is owned by this trial but cannot supply
    # independent identity evidence about other things moving in the scene.
    independent = not identified
    await hold(hands, run, hers, key, at, trying=independent)
    # A trial of resting begun while nothing is held holds nothing new, and the hold leaves no time of its own: the
    # trial begins now. Offline 2026-10-13 a game was ended in its first half minute by a rest asked for with nothing
    # held (None + duration).
    began = run.held_since if run.held_since is not None else at
    run.tried_at = began
    if counter_active:
        # A newly delivered phase can reserve the current observed cadence.
        # The previous input's lease and the programme's end do not move.
        run.counter_trial.phase_seconds = duration
    owner = calibration_owner(hers, hers.thing(moves)) if identified else None
    acquisition = owner is not None and not independent and hers.carried
    hard_s = maximum_s if acquisition else duration
    run.body_trial = ControlTrial(key, began, began + hard_s, epoch, owner, independent,
                                  began + duration if acquisition else None)
    lease_control_trial(hands, run, hers, run.body_trial, field='body_trial')
    run.longest_control_trial_s = max(getattr(run, "longest_control_trial_s", 0.), hard_s)
