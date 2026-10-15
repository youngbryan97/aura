"""Use a short steering prediction, observe it, and give calibration its next turn."""
from __future__ import annotations

import asyncio
import math
import re
import time
from typing import Any

from core.perception.motion_kept_between_pictures import (
    measured_motion_reference,
    measured_velocity,
)
from core.perception.position_from_the_same_points import measured_position
from core.perception.translation_from_visible_support import motion_was_measured
from core.runtime.task_ownership import fire_and_forget
from core.verify import invariant


def observed_steering_model(faces: Any, mine: Any, hers: Any, moves: Any, at: float, *, run: Any = None) -> None:
    """Feed current owned pixels and delivered input into a provisional physical model."""
    from core.agency.a_model_for_steering import ProvisionalSteeringBank

    model = getattr(faces, 'steering_model', None)
    if model is None:
        model = faces.steering_model = ProvisionalSteeringBank()
    owner = _owner(hers, mine)
    if owner is None:
        model.saw(None, '', at, at, 0., 0., 0., 0., measured=False)
        return
    receipt = hers._held_at(at)
    measured = (owner is not None and receipt is not None and motion_was_measured(mine, at)
                and at >= hers._beside_until)
    basis = bool(getattr(mine, 'motion_position', None) is not None)
    reference = measured_motion_reference(mine, at)
    if owner is not None and getattr(faces, 'steering_basis', reference) != reference:
        model = faces.steering_model = ProvisionalSteeringBank()
    faces.steering_basis = reference
    measured = measured and reference is not None
    if run is not None and owner is not None:
        model.arm(_turn_keys(run, faces), body_extent=max(mine.w, mine.h) if mine is not None else None)
    key, began = (receipt[1], receipt[0]) if receipt is not None else ('', at)
    x, y = measured_position(mine) if mine is not None else (0., 0.)
    vx, vy, velocity_error = measured_velocity(mine, at)
    motion = getattr(moves, '_point_motion', {}).get(getattr(mine, 'number', None))
    fit = getattr(motion, 'rigid', None)
    error = max(1., float(fit.error)) if basis and fit is not None else 1. if basis else 2.
    model.saw(owner, key, began, at, x, y, vx, vy, position_error=error, measured=measured,
              measured_velocity_error=velocity_error)


def _owner(hers: Any, mine: Any) -> tuple | None:
    if mine is None or hers.number is None or hers.kind is None or getattr(mine, 'number', None) != hers.number:
        return None
    born = getattr(mine, 'born', None)
    epoch = getattr(hers.identification, 'epoch', None)
    if (isinstance(born, bool) or not isinstance(born, (int, float)) or not math.isfinite(born)
            or type(epoch) is not int or epoch < 0):
        return None
    return mine.number, float(born), epoch


def _turn_keys(run: Any, faces: Any) -> set[str]:
    """Explicit body-turn roles or observed curved trials arm the structural prior."""
    from core.cognition.a_guide_to_a_place import THE_GUIDE

    guide = THE_GUIDE.get()
    controls = getattr(guide, 'controls', {}) or {}
    source = getattr(guide, 'code_says', None)
    named = set()
    for key in getattr(run, 'keys', ()):
        control = controls.get(key)
        action = ' '.join(str(getattr(control, field, '')) for field in ('act', 'on')).replace('_', ' ').replace('-', ' ')
        if re.search(r'\b(?:steer|rotate)\b|\bturn(?:s|ing)?\s+(?:left|right|clockwise|counterclockwise)\b', action, re.I):
            named.add(key)
        if _source_turns_body(source, key):
            named.add(key)
        if faces.travel_turns.curved(key) and not _source_drives_body(source, key):
            named.add(key)
    return named


def _source_body_method(source: Any, key: str) -> str | None:
    """Resolve the controlled receiver and method before assigning a structural role."""
    source_keys = getattr(source, 'keys', {}) or {}
    hers = getattr(source, 'hers', {}) or {}
    if not isinstance(source_keys, dict) or not isinstance(hers, dict):
        return None
    control = source_keys.get(key)
    owner = hers.get('class')
    if (not isinstance(owner, str) or not re.fullmatch(r'[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*', owner)
            or getattr(control, 'does', None) != 'moves'):
        return None
    by = getattr(control, 'by', None)
    if not isinstance(by, str):
        return None
    call = re.match(r'\s*([A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*)\.([A-Za-z_$][\w$]*)\s*\(', by)
    if call is None:
        return None
    receiver, method = call.groups()
    local = receiver in ('this', 'self') and str(getattr(control, 'where', '')).rsplit('.', 1)[0] == owner
    if receiver != owner and not local:
        return None
    return re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', method).replace('_', ' ').replace('$', ' ').lower()


def _source_turns_body(source: Any, key: str) -> bool:
    """A resolved body method name may arm a provisional rotation family."""
    words = _source_body_method(source, key)
    if words is None:
        return False
    if re.search(r'\bturn(?:s|ing)?\s+(?:on|off)\b|\b(?:music|audio|sound|mute|power|button|buttons|furniture)\b', words):
        return False
    if re.search(r'\b(?:get|read|is|has|can|test|check)\b', words):
        return False
    return bool(re.search(r'\b(?:turn(?:s|ing)?|rotat(?:e|es|ing|ion))\b', words))


def _source_drives_body(source: Any, key: str) -> bool:
    """A drive's curved path alone does not establish a rotation input.

    Acceleration may follow an already turned body or a collision. Offline
    2026-10-13 such curves removed both drive keys from provisional steering.
    This preserves a structural prior, never measured control status.
    """
    words = _source_body_method(source, key)
    if words is None or _source_turns_body(source, key):
        return False
    if re.search(r'\b(?:get|read|is|has|can|test|check|music|audio|sound|mute|power)\b', words):
        return False
    return bool(re.search(r'\b(?:accel(?:erate|erates|eration|erating)?|thrust|brak(?:e|es|ing)|propel|drive|drives)\b', words))


def _clearance(choosing: Any):
    mine = choosing.mine
    tall, wide = choosing.moves.shape
    through = getattr(choosing, 'way_through', None)

    def clear(x: float, y: float, error: float) -> bool:
        if not all(math.isfinite(v) for v in (x, y, error)) or error < 0:
            return False
        if not (mine.w / 2 + error <= x <= wide - mine.w / 2 - error
                and mine.h / 2 + error <= y <= tall - mine.h / 2 - error):
            return False
        cell = getattr(through, 'cell', 0.)
        if cell:
            margin = max(mine.w, mine.h) / 2 + error
            for row in range(int((y - margin) // cell), int((y + margin) // cell) + 1):
                for col in range(int((x - margin) // cell), int((x + margin) // cell) + 1):
                    if (col, row) in through.walls:
                        return False
        return True

    return clear


async def steer_with_a_model(hands: Any, run: Any, hers: Any, choosing: Any, at: float, *, hold: Any) -> bool:
    """Deliver a bounded provisional action between complete calibration experiments.

    Offline 2026-10-13 an owned body had useful moving observations but no
    fully calibrated steering rate. A physical hypothesis can choose a short
    action and check its prediction without establishing measured control.
    """
    faces, mine = getattr(run, 'faces', None), choosing.mine
    owner = _owner(hers, mine)
    model = getattr(faces, 'steering_model', None)
    active = getattr(run, '_modelled_steering', None)
    if active is not None:
        action, until, since, source_model = active
        same = (owner == action.owner and model is source_model and run.held == action.key and run.held_since == since)
        contradicted = model is not None and model.last_result in ('contradicted', 'partial contradicted')
        ended = math.isfinite(at) and max(at, time.monotonic()) >= until
        if same and not contradicted and math.isfinite(at) and not ended:
            return True
        deadline = getattr(run, '_steering_input_deadline', None)
        if deadline is not None and not deadline.done():
            deadline.cancel()
        run._model_offers_calibration = True
        if (not same or contradicted or ended) and run.held == action.key and run.held_since == since:
            from core.agency.inputs_while_measuring_control import release_control_if_owned

            if action.key:
                await release_control_if_owned(hands, run, hers, action.key, since)
            else:
                source_model.ended(action, since, max(at, time.monotonic()))
            run._modelled_steering = None
            return True
        run._modelled_steering = None
    if model is None or owner is None or hers.follows_pointer or not motion_was_measured(mine, at):
        return False
    trial = getattr(run, 'facing_trial', None)
    if (trial is not None and run.held_trying == trial.independent
            and trial.waits_for_picture(run.held, run.held_since, hers.identification.epoch, at)):
        return False
    if getattr(run, '_model_offers_calibration', False):
        run._model_offers_calibration = False
        return False
    if faces.turns_her():
        return False
    turn_keys = _turn_keys(run, faces)
    model.arm(turn_keys, body_extent=max(mine.w, mine.h))
    driver = faces.ahead_key(least=1) or model.driver(run.keys, now=time.monotonic())
    if not turn_keys or driver not in run.keys:
        return False
    (gx, gy), why, _aim = choosing.target()
    if why == 'wait' or gx is None and gy is None:
        return False
    x, y = measured_position(mine)
    target = (x if gx is None else gx, y if gy is None else gy)
    if math.dist((x, y), target) <= max(mine.w, mine.h) / 2:
        return False
    horizon = min(.35, max(.06, 2 * choosing.next_picture_s))
    now = time.monotonic()
    delay = max(0., choosing.response_s, now - at)
    action = model.action(run.keys, driver, target, horizon=horizon, delay_s=delay,
                          clearance=_clearance(choosing), now=now)
    if action is None:
        return False
    if (action.key and action.key not in run.keys or action.owner != owner or action.status != 'provisional'
            or not math.isfinite(action.duration) or not 0 < action.duration <= .35):
        return False
    if action.key == run.held and run.held_trying:
        await hold(hands, run, hers, '', at, trying=False)
    await hold(hands, run, hers, action.key, at, trying=False)
    if not model.delivered(action, run.held_since):
        await hold(hands, run, hers, '', at, trying=False)
        return True
    run.facing_trial = None
    run._modelled_steering = action, max(run.held_since, time.monotonic()) + action.duration, run.held_since, model
    if action.key:
        run._steering_input_deadline = fire_and_forget(
            _release_after_prediction(hands, run, hers, action.key, run.held_since, action.duration,
                                      until=run._modelled_steering[1]),
            name='provisional_steering_input_deadline', owner='realtime_control', bounded=True)
    return True


async def _release_after_prediction(hands: Any, run: Any, hers: Any, key: str, since: float, duration: float,
                                    *, until: float | None = None) -> None:
    """End a provisional press even when the next picture arrives late."""
    from core.agency.inputs_while_measuring_control import release_control_if_owned

    await asyncio.sleep(max(0., until - time.monotonic()) if until is not None else duration)
    await release_control_if_owned(hands, run, hers, key, since)


@invariant('agency.provisional_steering_keeps_input_ownership', scope='agency',
           owner='core/agency/steering_from_a_model.py', observational=False)
def _provisional_steering_input_invariant() -> tuple:
    from types import SimpleNamespace

    from core.cognition.a_guide_to_a_place import THE_GUIDE

    hers = SimpleNamespace(number=1, kind=0, identification=SimpleNamespace(epoch=3))
    mine = SimpleNamespace(number=1, born=0., w=10., h=10.)
    assert _owner(hers, mine) == (1, 0., 3)
    hers.number = None
    assert _owner(hers, mine) is None
    choosing = SimpleNamespace(mine=mine, moves=SimpleNamespace(shape=(100, 100)), way_through=None)
    assert _clearance(choosing)(20., 20., 2.) and not _clearance(choosing)(2., 20., 2.)
    run = SimpleNamespace(keys=['a', 'b'])
    faces = SimpleNamespace(travel_turns=SimpleNamespace(curved=lambda _key: False))
    controls = {'a':SimpleNamespace(act='turn_left', on=''), 'b':SimpleNamespace(act='turn on music', on='')}
    token = THE_GUIDE.set(SimpleNamespace(controls=controls))
    try:
        assert _turn_keys(run, faces) == {'a'}
    finally:
        THE_GUIDE.reset(token)
    return ()


@invariant('agency.source_turn_roles_only_arm_a_body_hypothesis', scope='agency',
           owner='core/agency/steering_from_a_model.py', observational=False)
def _source_turn_role_invariant() -> tuple:
    from types import SimpleNamespace

    source = SimpleNamespace(hers={'class':'Vehicle'}, keys={
        'a':SimpleNamespace(does='moves', by='Vehicle.mTurn(1)', where='Controls.update'),
        'b':SimpleNamespace(does='moves', by='Vehicle.turnOnMusic()', where='Controls.update'),
        'c':SimpleNamespace(does='moves', by='Furniture.rotateLeft()', where='Controls.update')})
    assert _source_turns_body(source, 'a')
    assert not _source_turns_body(source, 'b') and not _source_turns_body(source, 'c')
    return ()
