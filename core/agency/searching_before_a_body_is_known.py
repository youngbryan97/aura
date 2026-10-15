"""A finite pixel search follows an unresolved ordinary body experiment.

The search shares the actor's current capture and input lane. Its result
only changes point sampling on the exact last capture; ordinary observations
and new control trials must still establish a continuing body.
"""
from __future__ import annotations

import asyncio
import secrets
import time
from dataclasses import dataclass
from typing import Any

from core.agency.an_observed_scene_transition import keep_scene_transition
from core.agency.collecting_independent_pixel_trials import (
    PixelExperimentInputs,
    collect_independent_pixel_trials,
)
from core.agency.discovering_control_in_current_pixels import PixelResponseDiscovery
from core.agency.pixels_inside_control_trials import RandomizedPixelExperiment, freeze_frame
from core.agency.time_to_observe_a_control import observed_response_seconds
from core.perception.a_search_hint_for_these_pixels import make_search_hint, sampling_hint_matches
from core.perception.pixel_motion_before_object_identity import pixel_motion_signature
from core.perception.translation_from_visible_support import motion_was_measured
from core.runtime.executors import run_interactive_cpu
from core.verify import invariant


@dataclass
class PixelSearchState:
    active: bool = False
    attempted: bool = False
    status: str = "not attempted"
    trials: int = 0
    hint_applied: bool = False
    kept_priority_points: int | None = None
    fresh_priority_points: int | None = None
    release_confirmed: bool = True
    reference: str | None = None
    region: tuple | None = None
    keys: tuple[str, ...] = ()
    settle_s: float | None = None
    observed_frames: int = 0
    body_identified: bool = False
    phase_changed: bool = False


def search_inputs(hands: Any, run: Any, hers: Any) -> PixelExperimentInputs:
    state = run._pixel_search = PixelSearchState()
    return PixelExperimentInputs(hands, run, hers,
                                 owns=lambda: state.active and getattr(state, "may_collect", lambda: False)())


def _pair(keys: tuple[str, ...], preferred: tuple[str, ...] = ()) -> tuple[str, str] | None:
    candidates = tuple(dict.fromkeys([*(k for k in preferred if k in keys), *keys]))
    return (candidates[0], candidates[1]) if len(candidates) >= 2 else None


def eligible_pixel_search(run: Any, hers: Any, moves: Any, at: float, remaining: float) -> tuple[str, str] | None:
    state, how = getattr(run, "_pixel_search", None), getattr(run, "how", None)
    keys = tuple(dict.fromkeys(getattr(run, "keys", ())))
    trial = getattr(run, "body_trial", None)
    if (not isinstance(state, PixelSearchState) or state.attempted or state.active or remaining < 8.
            or how is None or how.has_a_body() is False or not (how.has_a_body() is True or how.body_keys_from_code)
            or hers.number is not None or hers.follows_pointer or getattr(run, "pointer_first", False)
            or getattr(run, "pointer_trigger", False) or how.against_someone()
            or not getattr(moves, "seeing", False) or len(keys) < 2
            or getattr(run, "trying", -1) < 4 * len(keys) or getattr(run, "held", "")
            or trial is not None and (trial.key or at < trial.until)
            or getattr(getattr(run, "shown", None), "asks", lambda _at: False)(at)
            or getattr(run, "_prepared_input_set", None) is not None
            or getattr(run, "_reflex_input", None) is not None
            or getattr(run, "letting_go", ()) or getattr(getattr(run, "charging", None), "key", "")):
        return None
    controls = tuple(k for k in (how.body_keys_from_code or how.keys_for_body_trials(keys)) if k in keys)
    return _pair(controls, (*getattr(how, 'translating_keys_from_code', ()),
                            *getattr(how, 'source_trial_keys', ())))


async def search_after_unresolved_body(hands: Any, run: Any, hers: Any, moves: Any, *,
        look: Any, context: Any, picture: Any, at: Any, deadline: float) -> bool:
    """Return True after an intervention; its preceding decision must be discarded."""
    pair = eligible_pixel_search(run, hers, moves, at, deadline - time.monotonic())
    origin = freeze_frame(picture, at)
    if pair is None or not isinstance(hands, PixelExperimentInputs) or origin is None:
        return False
    state = run._pixel_search
    state.settle_s = observed_response_seconds(getattr(run, 'intervals', ()), least_s=.35, settle_s=.2,
                                               samples=3, span_s=.15, longest_s=.8)
    if state.settle_s is None:
        state.status = 'observation cadence outside pixel trial budget'
        return False
    state.attempted, state.active = True, True
    state.keys = pair
    owner_task = asyncio.current_task()
    def may_collect():
        if owner_task.cancelling() or time.monotonic() >= deadline:
            return False
        context.check()
        return True
    state.may_collect = may_collect
    experiment = RandomizedPixelExperiment(pair, seed=secrets.randbits(64))
    discovery = PixelResponseDiscovery(experiment)
    latest = None
    pending_observation = None
    pending_observation_at = None
    async def fetch_owned_capture():
        nonlocal latest
        context.check()
        if time.monotonic() >= deadline:
            raise TimeoutError("the actor's capture deadline ended")
        latest = await look()
        context.check()
        current = freeze_frame(*latest) if isinstance(latest, tuple) and len(latest) == 2 else None
        if current is None or (current.surface, current.epoch, current.width, current.height) != (
                origin.surface, origin.epoch, origin.width, origin.height):
            raise RuntimeError("the pixel search's capture owner changed")
        return latest
    async def observe(frame):
        nonlocal pending_observation, pending_observation_at
        image, captured_at = latest
        if freeze_frame(image, captured_at) != frame:
            raise RuntimeError('the observed capture changed before measurement')
        # The normal eyes retain every current material path while this
        # experiment owns input. Their existing receipts still decide control.
        delivered = hands._receipts(), hands._interference, run.held, run.held_since, hers.identification.epoch
        pending_observation_at = captured_at
        pending_observation = asyncio.create_task(moves.see(image, captured_at))
        # Releasing a timed-out press must not cancel and retire the shared eyes.
        happened = await asyncio.shield(pending_observation)
        pending_observation = None
        context.check()
        if (not hands.owns() or delivered != (hands._receipts(), hands._interference, run.held,
                                             run.held_since, hers.identification.epoch)):
            raise RuntimeError('the input owner changed while observing its capture')
        if happened is not None:
            hers.saw(moves, happened, captured_at)
            if getattr(run, 'presses', None) is not None:
                run.presses.saw(moves.things, captured_at)
            previous = getattr(run, 'picture_at', None)
            if previous is not None and 0 < captured_at - previous <= .5:
                run.intervals.append(captured_at - previous)
            run.things, run.picture, run.picture_at = moves.things, image, captured_at
            run.pictures = getattr(run, 'pictures', 0) + 1
            if getattr(run, 'observation_sources', None) is not None:
                run.observation_sources['canvas-paint-v1' if getattr(image, 'drawing_scene', None) is not None else 'pixels'] += 1
            state.observed_frames += 1
            state.phase_changed = keep_scene_transition(run, happened, captured_at)
            mine = hers.thing(moves)
            state.body_identified = (mine is not None and mine.seen == captured_at
                                     and motion_was_measured(mine, captured_at))
    async def measure(press):
        context.check()
        signature = await run_interactive_cpu(pixel_motion_signature, press.before.rgb(), press.after.rgb(),
                                              tile=24, point_limit=256)
        context.check()
        return discovery.accept_last_press(press, signature)
    try:
        result = await collect_independent_pixel_trials(experiment, look=fetch_owned_capture, inputs=hands,
                                                        after_capture=observe, after_press=measure,
                                                        settle_s=state.settle_s, capture_s=.4, ack_s=.4,
                                                        sample_s=max(.02, (state.settle_s - .2) / 3),
                                                        stop_when=lambda: state.body_identified or state.phase_changed,
                                                        total_s=min(45., deadline - time.monotonic() - 1.))
        state.status, state.trials, state.release_confirmed = result.status, result.trials, result.release_confirmed
        if state.phase_changed:
            state.status = 'observed phase change'
        if not result.release_confirmed:
            raise RuntimeError("pixel search has an unconfirmed physical input release")
        if experiment.complete:
            region = await run_interactive_cpu(discovery.region)
            context.check()
            if region is not None and latest is not None:
                state.reference, state.region = region.reference, region.bounds
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return True
                try:
                    image, captured_at = await asyncio.wait_for(fetch_owned_capture(), min(.4, remaining))
                except TimeoutError:
                    state.status = 'fresh hint capture unavailable'
                    return True
                if captured_at.lower <= experiment.trials[-1].after.upper:
                    return True
                height, width = experiment.trials[-1].after.height, experiment.trials[-1].after.width
                left, top, right, bottom = region.sampling_bounds or region.bounds
                hint = make_search_hint(image, captured_at, normalized_region=(left / width, top / height,
                                                                             right / width, bottom / height))
                if hint is not None and time.monotonic() - captured_at.upper <= .5:
                    happened = await moves.see(image, captured_at, search_hint=hint)
                    receipt = getattr(moves, "sampling_hint_receipt", None)
                    state.hint_applied = happened is not None and sampling_hint_matches(receipt, hint)
                    if state.hint_applied:
                        state.kept_priority_points = receipt.kept_priority_points
                        state.fresh_priority_points = receipt.fresh_priority_points
                    if state.hint_applied and receipt.kept_priority_points > 0:
                        run.trying = 0
    finally:
        state.active = False
        if pending_observation is not None:
            # Input cleanup has already run. Join this bounded observation
            # before another phase can use its public tracker snapshot.
            finished, _pending = await asyncio.wait((pending_observation,), timeout=.6)
            if finished:
                happened = pending_observation.result()
                if happened is not None and keep_scene_transition(run, happened, pending_observation_at):
                    state.phase_changed = True
                    state.status = 'observed phase change'
            else:
                raise RuntimeError('pixel trial observation remains unresolved')
        if state.status == "not attempted":
            state.status = experiment.refused or "interrupted"
        state.trials = len(experiment.trials)
        state.release_confirmed = hands.release_problem is None and hands.pending_dispatch is None
    return True


def pixel_search_record(run: Any) -> dict[str, Any]:
    state = getattr(run, "_pixel_search", None)
    return ({name: getattr(state, name) for name in PixelSearchState.__dataclass_fields__}
            if isinstance(state, PixelSearchState) else {})


async def release_search_after_play(hands: Any, run: Any, hers: Any) -> None:
    """A pending search dispatch cannot make final actor cleanup wait without a bound."""
    from core.agency.inputs_while_measuring_control import release_after_play

    if isinstance(hands, PixelExperimentInputs) and (hands.release_problem or hands.pending_dispatch):
        if not await hands._finish(.4):
            from core.runtime.errors import record_degradation

            record_degradation("pixel_search_release", RuntimeError(hands.release_problem),
                               action="retained unresolved dispatch; stopped ordinary input")
            return
    await release_after_play(hands, run, hers)


@invariant("agency.pixel_search_follows_completed_unresolved_body_trials", scope="agency",
           owner="core/agency/searching_before_a_body_is_known.py", observational=False)
def _pixel_search_phase_invariant() -> tuple:
    from types import SimpleNamespace

    from core.agency.how_a_game_works import STEERED, Answer, HowItWorks

    run = SimpleNamespace(_pixel_search=PixelSearchState(), how=HowItWorks(me=Answer(STEERED, 1.)),
                          keys=["left", "right"], trying=7, held="", body_trial=None)
    hers, moves = SimpleNamespace(number=None, follows_pointer=False), SimpleNamespace(seeing=True)
    assert eligible_pixel_search(run, hers, moves, 10., 20.) is None
    run.trying = 8
    assert eligible_pixel_search(run, hers, moves, 10., 20.) == ("left", "right")
    hers.number = 3
    assert eligible_pixel_search(run, hers, moves, 10., 20.) is None
    return ()
