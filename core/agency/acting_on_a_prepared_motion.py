"""Deliver a prepared model action as an owned set of timed physical inputs.

A caller supplies a model and a state measured from the public observation.
This connects its retained path to the play actor. It does not estimate
unobserved channels, infer a model, or turn a predicted goal into a win.
"""
from __future__ import annotations

import asyncio
import copy
import logging
import math
import time
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from core.agency.following_a_prepared_path import FollowingPreparedPath, PreparedInput
from core.agency.inputs_while_measuring_control import (
    HandsWithControlReceipts,
    _control_delivery_lock,
    hold_control,
    primary_delivery,
    current_owned_control_trial,
)
from core.perception.translation_from_visible_support import motion_was_measured
from core.runtime.task_ownership import fire_and_forget
from core.verify import invariant

logger = logging.getLogger(__name__)


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


@dataclass(frozen=True)
class TimedKeys:
    keys: frozenset[str]
    duration_s: float

    def __post_init__(self) -> None:
        if (not isinstance(self.keys, frozenset) or len(self.keys) > 16
                or any(not isinstance(k, str) or not k or len(k) > 64 for k in self.keys)
                or not _finite(self.duration_s)
                or not 0 < self.duration_s <= 60.):
            raise ValueError("a prepared input needs a finite bounded physical hold")


@dataclass(frozen=True)
class ObservedMotion:
    """State from current public observations, with all model context identities."""

    state: Any
    captured_at: float
    owner: tuple[int, float, int]
    basis_id: Hashable
    goal_id: Hashable


@dataclass(frozen=True)
class PreparedMotionBinding:
    """An independently supplied model; callbacks read public state and do bounded arithmetic.

    The owner is (body number, birth time, identification epoch). Basis and
    goal identities cover their geometry, reference and constraints. A new
    goal or model needs a new binding and newly prepared path.
    """

    follower: FollowingPreparedPath
    model_id: Hashable
    owner: tuple[int, float, int]
    basis_id: Hashable
    goal_id: Hashable
    observe: Callable[[Any, Any, float], ObservedMotion | None]
    advance: Callable[[Any, Hashable], Any]
    outcome: Callable[[Any], bool | None]
    actions: Mapping[Hashable, TimedKeys]
    max_age_s: float = .25
    ack_wait_s: float = .5
    completion_slack_s: float = .1
    replay_budget: int = 128

    def __post_init__(self) -> None:
        if (not isinstance(self.follower, FollowingPreparedPath) or not self.actions or len(self.actions) > 32
                or any(value is None for value in (self.model_id, self.basis_id, self.goal_id))
                or any(not isinstance(action, TimedKeys) for action in self.actions.values())
                or not all(callable(f) for f in (self.observe, self.advance, self.outcome))
                or not all(_finite(v) for v in (self.max_age_s, self.ack_wait_s, self.completion_slack_s))
                or not 0 < self.max_age_s <= 5. or not 0 < self.ack_wait_s <= 5.
                or not 0 <= self.completion_slack_s <= 5.
                or type(self.replay_budget) is not int or not 1 <= self.replay_budget <= 1024):
            raise ValueError("a prepared motion needs bounded observations and delivery")
        hash(self.identity)
        if self.follower.model_identity != self.identity:
            raise ValueError("the prepared path belongs to different model context")
        object.__setattr__(self, "actions", MappingProxyType(dict(self.actions)))

    @property
    def identity(self) -> tuple:
        return self.model_id, self.owner, self.basis_id, self.goal_id


@dataclass(frozen=True)
class PreparedMotionProvider:
    """Supply a stable binding once current public observations establish its owner.

    A play stretch renumbers observation identities. The supplied provider
    can wait for that stretch's body instead of borrowing a previous owner.
    It returns None while qualification is unavailable.
    """

    provide: Callable[[Any, Any, float], PreparedMotionBinding | None]

    def __post_init__(self) -> None:
        if not callable(self.provide):
            raise ValueError("a prepared motion provider needs an observation callback")


@dataclass
class _InputSet:
    binding: PreparedMotionBinding
    offered: PreparedInput
    command: TimedKeys
    geometry: tuple
    previous_beside: float
    source: Any
    began: float | None = None
    until: float | None = None
    task: asyncio.Task | None = None
    cleanup: asyncio.Task | None = None
    deliveries: list[tuple] = field(default_factory=list)
    status: str = "delivering"


def _body_owner(hers: Any, moves: Any, at: float) -> tuple | None:
    body = hers.thing(moves)
    born, epoch = getattr(body, 'born', None), getattr(hers.identification, 'epoch', None)
    if (body is None or hers.kind is None or type(body.number) is not int
            or isinstance(born, bool) or not isinstance(born, (int, float)) or not math.isfinite(born)
            or type(epoch) is not int or epoch < 0 or not motion_was_measured(body, at)):
        return None
    return body.number, float(born), epoch


def _capture_bounds(at: float) -> tuple[float, float] | None:
    lower, upper = getattr(at, 'lower', at), getattr(at, 'upper', at)
    if (any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
            for v in (at, lower, upper)) or not lower <= at <= upper):
        return None
    return float(lower), float(upper)


def _observed(binding: PreparedMotionBinding, moves: Any, hers: Any, at: float) -> ObservedMotion | None:
    bounds, now = _capture_bounds(at), time.monotonic()
    if (bounds is None or not 0 <= now - bounds[0] <= binding.max_age_s or bounds[1] > now
            or _body_owner(hers, moves, at) != binding.owner):
        return None
    observed = binding.observe(moves, hers, at)
    if (not isinstance(observed, ObservedMotion) or observed.captured_at != at
            or (observed.owner, observed.basis_id, observed.goal_id) != (binding.owner, binding.basis_id, binding.goal_id)):
        return None
    return observed


def _geometry(moves: Any) -> tuple:
    return tuple(getattr(moves, 'shape', ())), float(getattr(moves, 'scale', 1.))


def _owns_key(run: Any, active: _InputSet, key: str) -> bool:
    return (getattr(run, '_prepared_input_set', None) is active
            and getattr(run, '_prepared_key_owners', {}).get(key) is active)


async def abort_prepared_motion(hands: Any, run: Any, hers: Any) -> bool:
    """An external input or owner departure must first end the exact retained set."""
    active = getattr(run, '_prepared_input_set', None)
    if active is None:
        return True
    if active.task is not None and active.task is not asyncio.current_task():
        active.task.cancel()
    active.binding.follower.invalidate()
    return await _cleanup(hands, run, hers, active)


async def _cleanup(hands: Any, run: Any, hers: Any, active: _InputSet) -> bool:
    if active.cleanup is None or active.cleanup.done():
        active.cleanup = fire_and_forget(_release(hands, run, hers, active, completed=False),
                name='incomplete_prepared_input_cleanup', owner='realtime_control', bounded=True)
    return await asyncio.shield(active.cleanup)


async def _release(hands: Any, run: Any, hers: Any, active: _InputSet, *, completed: bool) -> bool:
    async with _control_delivery_lock(run):
        if getattr(run, '_prepared_input_set', None) is not active:
            return True
        owners = run._prepared_key_owners
        with primary_delivery(hands):
            for key in sorted(k for k, owner in owners.items() if owner is active):
                if not _owns_key(run, active, key):
                    continue
                began = time.monotonic()
                try:
                    await asyncio.wait_for(hands.up(key), active.binding.ack_wait_s)
                except Exception as why:  # noqa: BLE001 - every attempted key still needs its own release
                    logger.warning("a prepared input release remains unconfirmed for %s: %s", key, why)
                    active.status = "release unconfirmed"
                    continue
                active.deliveries.append(('up', key, began, time.monotonic()))
                if _owns_key(run, active, key):
                    del owners[key]
        if any(owner is active for owner in owners.values()):
            active.binding.follower.invalidate()
            return False
        return _closed(run, hers, active, completed=completed)


def _closed(run: Any, hers: Any, active: _InputSet, *, completed: bool) -> bool:
    ended = time.monotonic()
    body = getattr(run, 'things', {}).get(hers.number)
    current = (hers.number, getattr(body, 'born', None), hers.identification.epoch)
    completed = (completed and current == active.binding.owner
                 and getattr(run, 'prepared_motion_binding', None) is active.source
                 and getattr(run, '_provided_motion_binding', None) is active.binding)
    accepted = active.binding.follower.finished(active.offered, at=ended, succeeded=completed)
    hers._beside_until = max(active.previous_beside, ended)
    hers.holding('', ended, trying=False)
    run.held_since, run.held_trying = ended, False
    run._prepared_input_set = None
    active.status = "completed" if accepted else "retired"
    receipts = getattr(run, '_prepared_input_receipts', [])
    receipts.append((active.binding.identity, active.command, active.began, ended, accepted,
                     tuple(active.deliveries)))
    run._prepared_input_receipts = receipts[-64:]
    return True


async def _expire(hands: Any, run: Any, hers: Any, active: _InputSet) -> None:
    await asyncio.sleep(max(0., active.until - time.monotonic()))
    await _release(hands, run, hers, active, completed=True)


async def _down(hands: Any, run: Any, active: _InputSet) -> None:
    with primary_delivery(hands):
        for key in sorted(active.command.keys):
            if not active.binding.follower.is_current(active.offered, time.monotonic()):
                raise TimeoutError("the observation expired while delivering prepared input")
            run._prepared_key_owners[key] = active  # cancellation can leave an attempted down physically delivered
            began = time.monotonic()
            await asyncio.wait_for(hands.down(key), active.binding.ack_wait_s)
            completed = time.monotonic()
            active.deliveries.append(('down', key, began, completed))
            run.input_key_downs[key] += 1


async def _start(hands: Any, run: Any, hers: Any, binding: PreparedMotionBinding,
                 offered: PreparedInput, command: TimedKeys, moves: Any, at: float) -> bool:
    receipts = hands if isinstance(hands, HandsWithControlReceipts) else HandsWithControlReceipts(hands, run, hers)
    receipts._other_input('prepared input', time.monotonic())
    await hold_control(hands, run, hers, '', at)
    receipts._other_input('prepared input', time.monotonic())
    active = _InputSet(binding, offered, command, _geometry(moves), getattr(hers, '_beside_until', -math.inf),
                       run.prepared_motion_binding)
    run._prepared_key_owners = {}
    run._prepared_input_set = active
    try:
        async with _control_delivery_lock(run):
            hers.pressed_beside(math.inf)
            await _down(hands, run, active)
            started = binding.follower.started(offered, at=time.monotonic(), duration=command.duration_s,
                                               slack=binding.completion_slack_s)
            if started is None:
                raise TimeoutError("the prepared dispatch expired before all inputs were acknowledged")
            active.began, active.until, active.status = started.began, started.until, "holding"
            timing = getattr(run, 'control_timing', None)
            if timing is not None:
                dispatched = active.deliveries[0][2] if active.deliveries else started.began
                run.control_delivery = timing.delivered(at, dispatched, started.began)
            active.task = fire_and_forget(_expire(hands, run, hers, active),
                    name='prepared_motion_input_deadline', owner='realtime_control', bounded=True)
    except BaseException:
        binding.follower.invalidate()
        await _cleanup(hands, run, hers, active)
        raise
    return True


async def act_on_prepared_motion(hands: Any, run: Any, moves: Any, hers: Any, at: float) -> bool:
    """Connect a supplied observed model to physical input; absent qualification leaves ordinary play available."""
    if getattr(run, '_prepared_input_set', None) is None and current_owned_control_trial(run, hers, at) is not None:
        return False
    source = getattr(run, 'prepared_motion_binding', None)
    binding = source.provide(moves, hers, at) if isinstance(source, PreparedMotionProvider) else source
    run._provided_motion_binding = binding
    active = getattr(run, '_prepared_input_set', None)
    observed = _observed(binding, moves, hers, at) if isinstance(binding, PreparedMotionBinding) else None
    if active is not None:
        verdict = binding.outcome(observed.state) if observed is not None else None
        viable = (active.binding is binding and observed is not None and active.geometry == _geometry(moves)
                  and active.status == 'holding'
                  and verdict is False)
        if not viable:
            await abort_prepared_motion(hands, run, hers)
        return True
    if observed is None or run.letting_go or run.charging.key:
        return False
    lower, _upper = _capture_bounds(at)
    offered = binding.follower.feedback(copy.deepcopy(observed.state), model_identity=binding.identity,
            observed_at=lower, now=time.monotonic(), max_age=binding.max_age_s,
            advance=binding.advance, outcome=binding.outcome, budget=binding.replay_budget)
    if offered is None:
        if binding.follower.status == 'observed_goal':
            await hold_control(hands, run, hers, '', at)
            return True
        return binding.follower.status == 'needs_observation' and binding.follower._finished_at >= lower
    command = binding.actions.get(offered.action)
    if command is None or not command.keys <= set(run.keys):
        binding.follower.invalidate()
        return False
    return await _start(hands, run, hers, binding, offered, command, moves, at)


@invariant('agency.prepared_motion_releases_only_its_owned_input_set', scope='agency',
           owner='core/agency/acting_on_a_prepared_motion.py', observational=False)
def _prepared_input_set_invariant() -> tuple:
    from types import SimpleNamespace

    old, fresh = object(), object()
    run = SimpleNamespace(_prepared_input_set=fresh, _prepared_key_owners={'a':fresh, 'b':old})
    assert _owns_key(run, fresh, 'a') and not _owns_key(run, old, 'a')
    assert not _owns_key(run, fresh, 'b') and not _owns_key(run, old, 'b')
    return ()


@invariant('agency.prepared_motion_requires_its_current_qualified_state', scope='agency',
           owner='core/agency/acting_on_a_prepared_motion.py', observational=False)
def _prepared_state_invariant() -> tuple:
    from types import SimpleNamespace

    from core.agency.solving_for_a_required_state import RequiredStatePlan
    from core.agency.which_one_answers_to_her import WhichIsHers

    at = time.monotonic()
    hers = WhichIsHers()
    hers.number, hers.kind = 1, 0
    body = SimpleNamespace(number=1, born=0., seen=at, x=0., motion_evidence='direct')
    moves = SimpleNamespace(things={1: body}, seeing=False)
    owner = 1, 0., hers.identification.epoch
    identity = 'measured model', owner, 'state coordinates', 'destination'
    follower = FollowingPreparedPath(RequiredStatePlan((0,), (0., 1.), True, 1, True), identity)
    binding = PreparedMotionBinding(follower, *identity,
        observe=lambda world, mine, captured: ObservedMotion(mine.thing(world).x, captured, owner, identity[2], identity[3]),
        advance=lambda state, _action: state + 1., outcome=lambda state: state == 1.,
        actions={0: TimedKeys(frozenset({'opaque'}), .03)})
    assert _observed(binding, moves, hers, at) is not None
    body.motion_evidence, body.motion_measured_at = 'unavailable', None
    assert _observed(binding, moves, hers, at) is None
    body.motion_evidence, body.born = 'direct', .1
    assert _observed(binding, moves, hers, at) is None
    return ()
