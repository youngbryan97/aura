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

from core.agency.collecting_independent_pixel_trials import (
    PixelExperimentInputs,
    collect_independent_pixel_trials,
)
from core.agency.discovering_control_in_current_pixels import PixelResponseDiscovery
from core.agency.pixels_inside_control_trials import RandomizedPixelExperiment, freeze_frame
from core.perception.a_search_hint_for_these_pixels import make_search_hint, sampling_hint_matches
from core.perception.pixel_motion_before_object_identity import pixel_motion_signature
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
    return _pair(controls, getattr(how, "translating_keys_from_code", ()))


async def search_after_unresolved_body(hands: Any, run: Any, hers: Any, moves: Any, *,
        look: Any, context: Any, picture: Any, at: Any, deadline: float) -> bool:
    """Return True after an intervention; its preceding decision must be discarded."""
    pair = eligible_pixel_search(run, hers, moves, at, deadline - time.monotonic())
    origin = freeze_frame(picture, at)
    if pair is None or not isinstance(hands, PixelExperimentInputs) or origin is None:
        return False
    state = run._pixel_search
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
    async def capture():
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
    async def measure(press):
        context.check()
        signature = await run_interactive_cpu(pixel_motion_signature, press.before.rgb(), press.after.rgb(),
                                              tile=24, point_limit=256)
        context.check()
        return discovery.accept_last_press(press, signature)
    try:
        result = await collect_independent_pixel_trials(experiment, look=capture, inputs=hands, after_press=measure,
                                                        capture_s=.4, ack_s=.4,
                                                        total_s=min(20., deadline - time.monotonic() - 1.))
        state.status, state.trials, state.release_confirmed = result.status, result.trials, result.release_confirmed
        if not result.release_confirmed:
            raise RuntimeError("pixel search has an unconfirmed physical input release")
        if experiment.complete:
            region = await run_interactive_cpu(discovery.region)
            context.check()
            if region is not None and latest is not None:
                state.reference, state.region = region.reference, region.bounds
                image, captured_at = latest
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
