"""Imagining the next moment of a world under each thing she could do, and doing the one that comes out best.

A finite game she can solve outright: she holds its rules exactly and searches
them (2048, a board in turns). A world that moves has no rules written down,
but she learns its parts as she plays: how each key moves her
(core/agency/which_one_answers_to_her.py), the path a press takes her on
(core/agency/what_a_press_does.py), how each kind of thing falls, bounces and
leaves (core/perception/how_things_move_here.py), and what meeting each kind
does to her (core/agency/what_meeting_things_does.py). Put together, those are
a simulator of the world as she knows it. This runs it.

For each plan she could follow over the next second or so (hold a key, hold
one then another, hold one and make a press partway), her thing is run
forward under it and every thing that matters is run forward under the world's
own going. The plan is scored by what it meets: what is worth getting, the
sooner the better; what costs her, heavily, and wider of it the lower she is;
and how near it ends to where she means to be. She does the first part of the
best plan, and plans again at the next picture: model-predictive control, as
the general video game players and robots do it, on a model she learned
herself.

What this replaces is a set of one-step rules, one per kind of play (dodge,
catch, intercept, jump over). A plan found by imagining needs no rule for its
kind of play: going left and then up to get under a falling thing past a wall,
or jumping while running so as to land beyond what is coming, comes out of the
imagining. Nothing here knows a game.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.agency.arriving_by_her_pushes import arrival_excess, arrival_holds, braking_pulse_holds, stopping_distance
from core.agency.a_requested_region_in_a_plan import RelationForecast, RelationInPlan, current_requested_region

__all__ = ["HORIZON_S", "Plan", "best_plan", "plans_to_imagine"]

#: How far ahead she imagines, and in what steps, in seconds.
HORIZON_S = 1.2
STEP_S = 1 / 30
#: When in a plan she may change from one key to another, or make a press, in seconds from now.
CHANGES_AT = (0.15, 0.3, 0.5, 0.75)
PRESSES_AT = (0.0, 0.2, 0.4)
#: What a thing worth getting is worth, what a thing that costs her costs, and how much each step of waiting takes off.
GAIN = 1.0
LOSS = 4.0
SOONER = 0.97
#: How much ending far from where she means to be costs, for the whole width of what she can cross in the horizon.
FAR_FROM_THE_GOAL = 0.5
#: How much a plan that changes what she is holding now costs: her hands are not free.
CHANGING = 0.02
#: How much waiting before a plan starts doing what it does costs, a second of it: her model of the world is never
#: exact, and of two plans that come out alike, the one that moves sooner leaves room for what it gets wrong.
WAITING = 0.2
#: Of a carried body's greatest push, the share of a second's worth of it under which it comes onto a thing gently,
#: where how fast meeting a thing may be is not known yet.
GENTLE_SHARE = 0.25

#: What a cell of ground still to be gone over is worth, where the place's code names its ground
#: (core/agency/ground_her_code_names.py): a quarter of a thing worth getting, so that a plan sweeping new ground
#: outweighs one that stands, and a cell is never worth meeting what costs her.
A_CELL_GONE_OVER = 0.25

#: How long she takes to answer, in seconds, from which a thing that comes for her is imagined turning toward her as
#: fast as it has been seen to turn, not going on as it goes. Measured on the seventh and twentieth levels of
#: Camp Lazlo: Paintcan Panic from its code (tests/fixtures/worlds_that_move/paintcan.html), seeds 1 to 12: answering
#: in 0.67 s, she won 6 imagining its chasers turning and 1 imagining them going straight on; answering in 0.25 s, 11
#: and 12. Imagined pushed as hard as seen and held back by drag, she won 3 and 9.
CHASED_WHERE_ANSWERING_S = 0.3

#: The most things imagined: the nearest that matter.
MOST_THINGS = 16
#: How much faster and slower than she has measured the world is imagined going, plan by plan: her measure of it is
#: never exact, and a plan that comes out well only at exactly the pace measured is no plan to stake a life on. A plan
#: worth following is one that comes out well however the world goes within that; this is what makes her get under a
#: falling thing and wait, as people do, rather than pass under it at the moment it arrives.
PACES = (0.85, 1.0, 1.15)


@dataclass(frozen=True)
class Plan:
    """What she does over the horizon: keys held in turn, each from its start, and presses made at times."""

    held: tuple[tuple[float, str], ...]
    presses: tuple[tuple[float, str], ...] = ()

    @property
    def first(self) -> str:
        return self.held[0][1]

    def press_now(self) -> str | None:
        return next((key for when, key in self.presses if when <= 0.0), None)


@dataclass
class Imagined:
    """What a plan came to: its worth, and what it met."""

    plan: Plan
    worth: float
    gets: int = 0
    costs: int = 0
    ends_at: tuple[float, float] = (0.0, 0.0)
    #: How many cells of ground not yet gone over it goes over.
    covers: int = 0
    #: Whether the plans imagined differ in what they meet: where none does, imagining has nothing to choose between.
    decides: bool = False
    relation: RelationForecast | None = None


@dataclass
class _Scene:
    """Her thing and the world, run forward once for every plan to be held against."""

    times: np.ndarray
    start: tuple[float, float]
    size: tuple[float, float]
    bounds: tuple[float, float, float, float]
    ways: dict[str, tuple[float, float]]
    rest: tuple[float, float]
    paths: dict[str, Callable[[float], tuple[float, float]]]
    #: Each thing's place at each step (paces x steps x things x 2), its size, what it is to her, and whether it is there.
    where: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0, 2)))
    sizes: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    worth: np.ndarray = field(default_factory=lambda: np.zeros(0))
    there: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0), dtype=bool))
    barred: Any = None
    #: For a body carried on by its own going: how much each key ("" for none) adds to its going each second, its going
    #: now, and the speed above which meeting each thing it means to come onto breaks on it (by thing).
    pushes: dict[str, tuple[float, float]] | None = None
    going: tuple[float, float] = (0.0, 0.0)
    gentle: np.ndarray = field(default_factory=lambda: np.zeros(0))
    arriving: bool = False
    arrival_near: tuple[float, float] = (0., 0.)
    arrival_gentle: float = 0.
    arrival_goal: tuple = (None, None)
    relation: RelationInPlan | None = None
    relation_bounds: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 4)))
    relation_velocity: tuple[float, float] = (0., 0.)
    delay: float = 0.
    #: The ground its code names, laid on her picture, with what she has gone over of it.
    ground: Any = None
    #: How far each thing could swerve off its going in a second, and how long she takes to answer
    #: (core/agency/how_far_a_thing_could_swerve.py).
    swerves: np.ndarray = field(default_factory=lambda: np.zeros(0))
    answer_s: float = 0.0
    #: Which things come for her (core/agency/what_comes_for_her.py), where each is once her keys reach the world, and
    #: how fast it goes: each is run toward where she is, plan by plan, not along its line.
    chasing: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    chase_from: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    chase_going: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    #: How fast each turns its going toward her (pixels a second, each second; infinite where it has not been seen
    #: turning), and the fastest it goes.
    chase_turn: np.ndarray = field(default_factory=lambda: np.zeros(0))
    chase_top: np.ndarray = field(default_factory=lambda: np.zeros(0))


def plans_to_imagine(choices: Sequence[str], presses: Sequence[str], held: str, *, every: float = 0.0) -> list[Plan]:
    """Every plan worth imagining: each key (or none) held throughout; each held and then another from a change time;
    and each press made at a time while a key is held. ``choices`` includes "" (nothing held).

    Nothing in a plan changes before her next decision, ``every`` seconds from now: a plan that changes keys sooner is
    one she cannot follow. Offline 2026-10-13, deciding three times a second on the paint world's own code, she chose
    plans that held a key for a sixth of a second and then turned, held each for a third, and went right and left in
    turn on the spot while a bug came to her: she lost every life on six seeds of six."""
    def kept(times: Sequence[float]) -> list[float]:
        return sorted({max(when, every) if when > 0.0 else when for when in times})

    plans = [Plan(((0.0, key),)) for key in choices]
    for first in choices:
        for then in choices:
            if then != first:
                plans += [Plan(((0.0, first), (when, then))) for when in kept(CHANGES_AT)]
    for press in presses:
        for key in choices:
            plans += [Plan(((0.0, key),), ((when, press),)) for when in kept(PRESSES_AT)]
    # The key held now first, so that among plans that come out alike she goes on as she is.
    return sorted(plans, key=lambda plan: plan.first != held)


def best_plan(choosing: Any, held: str, goal: tuple[float | None, float | None], *,
              horizon: float = HORIZON_S, arrive_at: Any = None) -> Imagined | None:
    """The plan that comes out best for ``choosing`` (the real-time player's arithmetic of one decision), holding
    ``held`` now and meaning to be at ``goal``; None where there is nothing to imagine with: no thing of hers, or no way
    she knows of moving it."""
    scene = _the_scene(choosing, horizon, goal=goal)
    if scene is None:
        return None
    plans = plans_to_imagine(["", *scene.ways], list(scene.paths), held,
                             every=float(getattr(choosing, "next_picture_s", 0.0) or 0.0))
    if scene.relation is not None and scene.pushes is not None:
        cadence = max(STEP_S, float(getattr(choosing, 'next_picture_s', STEP_S)))
        plans.extend(Plan(sequence) for sequence in braking_pulse_holds(scene.pushes, cadence, horizon))
    arrival = None
    if arrive_at is not None and scene.pushes is not None:
        cadence = max(STEP_S, float(getattr(choosing, "next_picture_s", STEP_S)))
        near = ((scene.size[0] + arrive_at.w) / 2, (scene.size[1] + arrive_at.h) / 2)
        gentle = choosing.meeting.gentle_below(arrive_at.kind) if hasattr(choosing.meeting, "gentle_below") else None
        if gentle is None:
            gentle = GENTLE_SHARE * max(math.hypot(*p) for p in scene.pushes.values())
        arrival_goal = goal
        if scene.relation is not None:
            arrival_goal = scene.relation.nearest(scene.start, scene.delay)
            # The centre region already accounts for both extents. Aim to
            # settle there; zero desired speed asserts no safe-contact limit.
            near, gentle = (0., 0.), 0.
        arrival = (arrival_holds(scene.start, scene.going, arrival_goal, scene.pushes, scene.times.tolist(), cadence, held,
                                 near=near, gentle=gentle) if arrival_goal is not None else None)
        if arrival:
            scene.arriving = True
            scene.arrival_near, scene.arrival_gentle = near, gentle
            scene.arrival_goal = arrival_goal
            plans.append(Plan(arrival))
    imagined = [_run(scene, plan, goal, held, choosing.caution) for plan in plans]
    best = max(imagined, key=lambda one: one.worth)
    if scene.relation is not None:
        # Round-off in a nominally stopped velocity must not buy another
        # push/brake cycle. This is a score tie, not a geometric tolerance.
        tolerance = 64 * len(scene.times) * math.ulp(max(1., abs(float(best.worth))))
        tied = [one for one in imagined if abs(float(one.worth-best.worth)) <= tolerance
                and (one.gets, one.costs, one.covers) == (best.gets, best.costs, best.covers)]
        best = min(tied, key=lambda one: (len(one.plan.held)+len(one.plan.presses), one.plan.first != held))
    best.decides = bool(arrival) or scene.relation is not None or len({(one.gets, one.costs, one.covers) for one in imagined}) > 1
    return best


def _the_scene(choosing: Any, horizon: float, *, goal: tuple | None = None) -> _Scene | None:
    mine = getattr(choosing, "mine", None)
    ways = dict(getattr(choosing, "ways", {}) or {})
    # Nothing to imagine with: no thing of hers, no way she knows of moving it, or no world seen around it.
    if mine is None or not ways or getattr(choosing, "moves", None) is None or not hasattr(choosing, "others"):
        return None
    tall, wide = choosing.moves.shape
    times = np.arange(1, int(round(horizon / STEP_S)) + 1) * STEP_S
    moving = ways.get(getattr(choosing, "held_now", ""), (0.0, 0.0))
    response = float(getattr(choosing, "response_s", 0.0) or 0.0)
    # What she holds now goes on moving her while the next command is on its way.
    start = (mine.x + moving[0] * response, mine.y + moving[1] * response)
    paths: dict[str, Callable[[float], tuple[float, float]]] = {}
    presses = getattr(choosing, "press_paths", None)
    for key in getattr(choosing, "lifts", []) or []:
        path = presses(key) if callable(presses) else None
        if path is not None:
            paths[key] = path
    scene = _Scene(times=times, start=start, size=(float(mine.w), float(mine.h)),
                   bounds=(mine.w / 2, mine.h / 2, wide - mine.w / 2, tall - mine.h / 2), ways=ways,
                   rest=tuple(getattr(choosing, "rest", (0.0, 0.0))), paths=paths,
                   barred=getattr(choosing, "barred", None), delay=response, ground=getattr(choosing, "ground", None))
    hers = getattr(choosing, "hers", None)
    if hers is not None and getattr(hers, "carried", False):
        # Carried on by its own going, a key adds to her going rather than setting it.
        pushes = {key: hers.push_of(key) for key in ["", *ways]}
        scene.pushes = {key: push for key, push in pushes.items() if push is not None}
        scene.going = (float(mine.vx), float(mine.vy))
        scene.start = (float(mine.x), float(mine.y))
        if response:
            held = getattr(choosing, "held_now", "")
            if held not in scene.pushes:
                return None
            ax, ay = scene.pushes[held]
            scene.start = (mine.x + mine.vx * response + .5 * ax * response ** 2,
                           mine.y + mine.vy * response + .5 * ay * response ** 2)
            scene.going = (mine.vx + ax * response, mine.vy + ay * response)
    region = current_requested_region(choosing, goal)
    if region is not None:
        target = choosing.goal_binding.thing
        path = _its_path(getattr(choosing, 'physics', None), target, horizon * max(PACES) + response)
        relation = RelationInPlan(region, (float(target.x), float(target.y)), path)
        bands = [[relation.bounds_after(response + float(when) * pace) for when in times] for pace in PACES]
        if all(bounds is not None for band in bands for bounds in band) and len(times) >= 2:
            scene.relation = relation
            scene.relation_bounds = np.asarray(bands)
            dt = float(times[-1] - times[-2])
            last = PACES.index(1.)
            scene.relation_velocity = tuple((scene.relation_bounds[last, -1, :2]
                                             - scene.relation_bounds[last, -2, :2]) / dt)
    _the_world_run_forward(choosing, scene)
    return scene


def _the_world_run_forward(choosing: Any, scene: _Scene) -> None:
    """Every thing that matters to her, nearest first, run forward under the world's own going."""
    from core.agency.what_meeting_things_does import AVOID, CLICK, MEET, SHOOT

    worth_of = {MEET: GAIN, CLICK: GAIN, AVOID: -LOSS, SHOOT: -LOSS}
    # A kind that costs her as much getting by her as touching her is nothing to keep clear of (what_meeting_things_does).
    anyway = getattr(getattr(choosing, "meeting", None), "lands_anyway", ())
    things = [(t, 0.0 if int(t.kind) in anyway else worth_of.get(choosing.stance(t), 0.0)) for t in choosing.others()]
    binding = getattr(choosing, 'goal_binding', None)
    if (binding is not None and getattr(getattr(binding, 'request', None), 'relation', None) in ('on', 'inside')
            and getattr(getattr(choosing, 'meeting', None), 'spatial_request', None) == binding.request):
        # Overlap with a requested region supplies no contact reward. Keep
        # independently known danger and all other earned interactions.
        things = [(t, w) for t, w in things if w <= 0. or t is not binding.thing]
    things = [(t, w) for t, w in things if w != 0.0]
    physics = getattr(choosing, "physics", None)
    # What she plans with is a construal of the scene, as people's is (value-guided construal, Ho et al. 2022): what
    # cannot come within her reach in the time imagined cannot change her route, and is left out before planning; of
    # the rest, what could reach her soonest is imagined first. A fast thing from far off outranks a near one going away.
    fastest = max((math.hypot(*way) for way in scene.ways.values()), default=0.0)
    construed = []
    for thing, worth in things:
        path = _its_path(physics, thing, float(scene.times[-1]) * max(PACES) + scene.delay)
        if scene.delay:
            def delayed(after, original=path):
                return original(after + scene.delay)
            path = delayed
        soonest = _soonest_within_reach(path, scene, fastest, thing)
        if soonest is not None:
            construed.append((soonest, thing, worth, path))
    construed.sort(key=lambda one: one[0])
    construed = construed[:MOST_THINGS]
    things = [(thing, worth) for _soonest, thing, worth, _path in construed]
    steps, count = len(scene.times), len(things)
    where = np.zeros((len(PACES), steps, count, 2))
    there = np.ones((len(PACES), steps, count), dtype=bool)
    for column, (_soonest, _thing, _worth, path) in enumerate(construed):
        for layer, pace in enumerate(PACES):
            for row, when in enumerate(scene.times):
                spot = path(float(when) * pace)
                if spot is None:
                    there[layer, row:, column] = False
                    break
                where[layer, row, column] = spot
    scene.where, scene.there = where, there
    scene.sizes = np.array([[float(t.w), float(t.h)] for t, _w in things]).reshape(count, 2)
    scene.worth = np.array([w for _t, w in things])
    swerve = getattr(choosing, "swerve", None)
    scene.swerves = np.array([float(swerve(t)) if callable(swerve) else 0.0 for t, _w in things])
    scene.answer_s = float(choosing.answer_s()) if hasattr(choosing, "answer_s") else 0.0
    # Where she answers slowly, what comes for her turns toward her before she can see it do so, and is imagined
    # turning; where quickly, what it does next is seen in time, and imagined going on as it goes.
    comes = getattr(choosing, "comes", None) if scene.answer_s >= CHASED_WHERE_ANSWERING_S else None
    scene.chasing = np.array([bool(comes is not None and comes.comes_for_her(t, max(scene.size))) for t, _w in things],
                             dtype=bool).reshape(count)
    scene.chase_from = np.array([[float(t.x) + float(t.vx) * scene.delay, float(t.y) + float(t.vy) * scene.delay]
                                 for t, _w in things]).reshape(count, 2)
    scene.chase_going = np.array([[float(t.vx), float(t.vy)] for t, _w in things]).reshape(count, 2)
    from core.agency.what_comes_for_her import how_it_turns

    turning = [how_it_turns(getattr(t, "path", None)) if chasing else None
               for (t, _w), chasing in zip(things, scene.chasing, strict=True)]
    scene.chase_turn = np.array([turn[0] if turn else math.inf for turn in turning]).reshape(count)
    scene.chase_top = np.array([max(turn[1], math.hypot(float(t.vx), float(t.vy))) if turn
                                else math.hypot(float(t.vx), float(t.vy)) for turn, (t, _w) in zip(turning, things, strict=True)
                                ]).reshape(count)
    if scene.pushes is not None:
        # What she comes onto, she comes onto gently: under the speed meeting it has been found to cost above, or a
        # gentle share of her going where that is not known yet.
        meeting = choosing.meeting
        fastest = max([math.hypot(*push) for push in scene.pushes.values()] + [1.0])
        gentle = [meeting.gentle_below(t.kind) if hasattr(meeting, "gentle_below") else None for t, _w in things]
        scene.gentle = np.array([g if g is not None else GENTLE_SHARE * fastest for g in gentle])


def _soonest_within_reach(path: Callable[[float], tuple[float, float] | None], scene: _Scene, fastest: float,
                          thing: Any) -> float | None:
    """The soonest time in the horizon a thing could be where she could be by then, going as fast as she goes; None
    where it never could."""
    near = (max(scene.size) + max(float(thing.w), float(thing.h))) / 2
    for when in scene.times:
        for pace in PACES:
            spot = path(float(when) * pace)
            if spot is not None and math.hypot(spot[0] - scene.start[0], spot[1] - scene.start[1]) <= (
                    fastest * float(when) + near):
                return float(when)
    return None


def _its_path(physics: Any, thing: Any, seconds: float) -> Callable[[float], tuple[float, float] | None]:
    """Where a thing will be after so many seconds, under the world's physics as learned where they are known, else
    along its going; None once it has left."""
    imagined = None
    if physics is not None and getattr(thing, "moved", False):
        try:
            imagined = physics.imagine(thing, seconds, step=STEP_S)
        except (AttributeError, TypeError, ValueError, ZeroDivisionError):
            imagined = None
    if imagined is None or not getattr(imagined, "path", None):
        return lambda after: thing.where_at(after)
    points = imagined.path
    leaves = getattr(imagined, "ends", "") == "leaves"
    last = points[-1][0]

    def at(after: float) -> tuple[float, float] | None:
        if after > last:
            return None if leaves else (points[-1][1], points[-1][2])
        index = min(len(points) - 1, max(0, int(round(after / STEP_S))))
        return points[index][1], points[index][2]

    return at


def _run(scene: _Scene, plan: Plan, goal: tuple[float | None, float | None], held: str, caution: float) -> Imagined:
    """Her thing run under ``plan``, held against the world as it goes, and what that comes to."""
    times = scene.times
    xs, ys = np.empty(len(times)), np.empty(len(times))
    speeds = np.zeros(len(times))
    x, y = scene.start
    low_x, low_y, high_x, high_y = scene.bounds
    previous = 0.0
    going = scene.going
    for index, when in enumerate(times):
        key = next(key for start, key in reversed(plan.held) if start <= when - STEP_S / 2 or start == 0.0)
        step = float(when) - previous
        if scene.pushes is not None:
            push = scene.pushes.get(key, scene.pushes.get("", (0.0, 0.0)))
            going = (going[0] + push[0] * step, going[1] + push[1] * step)
            vx, vy = going
        else:
            vx, vy = scene.ways.get(key, scene.rest) if key else scene.rest
        speeds[index] = math.hypot(vx, vy)
        previous = float(when)
        nx, ny = x + vx * step, y + vy * step
        if not _barred(scene.barred, nx, ny):
            x, y = nx, ny
        x = min(max(x, low_x), high_x) if high_x >= low_x else x
        y = min(max(y, low_y), high_y) if high_y >= low_y else y
        xs[index], ys[index] = x, y
    for at, press in plan.presses:
        path = scene.paths.get(press)
        if path is None:
            continue
        for index, when in enumerate(times):
            if when >= at:
                dx, dy = path(float(when) - at)
                xs[index] += dx
                ys[index] += dy
    worth, gets, costs = 0.0, 0, 0
    if scene.worth.size:
        # The plan held against the world at each pace, and what it comes to taken over all of them.
        for layer in range(len(PACES)):
            met, got, lost = _what_it_meets(scene, layer, xs, ys, caution, speeds,
                                            _chased(scene, layer, xs, ys) if scene.chasing.any() else None)
            worth += met / len(PACES)
            gets, costs = (got, lost) if PACES[layer] == 1.0 else (gets, costs)
    covers = 0
    if scene.ground is not None:
        met, covers = _ground_gone_over(scene.ground, xs, ys)
        worth += met
    gx, gy = goal
    relation_forecast = None
    if gx is not None or gy is not None:
        reach = max(1.0, max((math.hypot(*way) for way in scene.ways.values()), default=1.0) * float(times[-1]))
        if scene.arriving:
            reach = max(reach, math.hypot((scene.start[0] - gx) if gx is not None else 0.,
                                         (scene.start[1] - gy) if gy is not None else 0.))
        far = math.hypot((xs[-1] - gx) if gx is not None else 0.0, (ys[-1] - gy) if gy is not None else 0.0)
        if scene.relation is not None:
            bands = scene.relation_bounds
            dx = np.maximum(0., np.maximum(bands[:, :, 0]-xs, xs-bands[:, :, 2]))
            dy = np.maximum(0., np.maximum(bands[:, :, 1]-ys, ys-bands[:, :, 3]))
            distances = np.hypot(dx, dy)
            far = float(distances[:, -1].mean())
            # Track the region throughout the forecast. An endpoint alone
            # can prefer a detour that only returns at the horizon's end.
            worth -= FAR_FROM_THE_GOAL * min(2., float(distances.mean()) / reach)
            relative_speed = math.hypot(vx-scene.relation_velocity[0], vy-scene.relation_velocity[1])
            worth -= FAR_FROM_THE_GOAL * min(2., relative_speed * float(times[-1]) / reach)
            relation_forecast = RelationForecast(scene.relation.region, far, float(distances.mean()), relative_speed)
        worth -= FAR_FROM_THE_GOAL * min(2.0, far / reach)
        if scene.arriving and scene.pushes is not None:
            # A plan ending near the place at speed still has to stop. Give
            # it the cost of half that remaining room beyond the horizon.
            room = stopping_distance(going, scene.arrival_goal, scene.pushes)
            worth -= FAR_FROM_THE_GOAL * .5 * room / reach
            excess = arrival_excess((float(xs[-1]), float(ys[-1])), going, scene.arrival_goal, scene.pushes,
                                    scene.arrival_near, scene.arrival_gentle)
            worth -= LOSS * excess / max(1., scene.arrival_gentle)
    # A fixed switch tax can leave a current geometric constraint unmet.
    # Region tracking already scores waiting, and the scene owns delivery
    # delay. Stable plan ordering still resolves equal alternatives.
    if scene.relation is None:
        if plan.first != held:
            worth -= CHANGING
        if not plan.first and len(plan.held) > 1:
            worth -= WAITING * plan.held[1][0]
    return Imagined(plan, worth, gets, costs, (float(xs[-1]), float(ys[-1])), covers, relation=relation_forecast)


def _chased(scene: _Scene, layer: int, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Where each thing is at each step, at one pace, with those that come for her turned toward where her path has her
    then, as fast as each has been seen to turn and no faster than it has been seen to go: one seen turning at once
    goes straight at her and no further."""
    where = scene.where[layer].copy()
    pace = PACES[layer]
    for column in np.flatnonzero(scene.chasing):
        x, y = scene.chase_from[column]
        vx, vy = scene.chase_going[column] * pace
        turn, top = float(scene.chase_turn[column]) * pace, float(scene.chase_top[column]) * pace
        previous = 0.0
        for row, when in enumerate(scene.times):
            step = float(when) - previous
            previous = float(when)
            dx, dy = float(xs[row]) - x, float(ys[row]) - y
            far = math.hypot(dx, dy)
            if not math.isfinite(turn):
                go = min(far, top * step)
                x, y = (x + dx / far * go, y + dy / far * go) if far > 1e-9 else (x, y)
            else:
                if far > 1e-9:
                    vx, vy = vx + dx / far * turn * step, vy + dy / far * turn * step
                speed = math.hypot(vx, vy)
                if speed > top > 0.0:
                    vx, vy = vx * top / speed, vy * top / speed
                x, y = x + vx * step, y + vy * step
            where[row, column] = (x, y)
    return where


def _ground_gone_over(ground: Any, xs: np.ndarray, ys: np.ndarray) -> tuple[float, int]:
    """What a path's going over ground not yet gone over is worth, each new cell the sooner the better, and how many."""
    worth, new = 0.0, set()
    for index, (x, y) in enumerate(zip(xs, ys, strict=True)):
        cell = ground.cell_of(*ground.mark_of(float(x), float(y)))
        if cell is not None and cell not in ground.marked and cell not in new:
            new.add(cell)
            worth += A_CELL_GONE_OVER * (SOONER ** index)
    return worth, len(new)


def _what_it_meets(scene: _Scene, layer: int, xs: np.ndarray, ys: np.ndarray, caution: float,
                   speeds: np.ndarray | None = None, where: np.ndarray | None = None) -> tuple[float, int, int]:
    """What her path meets in the world run at one pace: its worth, how many worth getting it gets, how many cost her.
    A carried body coming onto a thing faster than it can be met breaks on it: that meeting costs. ``where`` is where
    each thing is at each step, where it is not the world's own going (things that come for her)."""
    where, there = (scene.where[layer] if where is None else where), scene.there[layer]
    apart_x = np.abs(where[:, :, 0] - xs[:, None])
    apart_y = np.abs(where[:, :, 1] - ys[:, None])
    reach_x = (scene.sizes[:, 0][None, :] + scene.size[0]) / 2
    reach_y = (scene.sizes[:, 1][None, :] + scene.size[1]) / 2
    costly = scene.worth < 0
    # What costs her is given a berth, the wider the lower she is; and as wide again as it could swerve off its going in
    # the time before she can answer it, which no plan made now can see coming.
    berth = np.where(costly, 2.0 * caution, 0.0)[None, :]
    if scene.swerves.size == costly.size and scene.swerves.any():
        unsure = np.minimum(scene.times, scene.answer_s)[:, None] * scene.swerves[None, :]
        berth = berth + np.where(costly[None, :], unsure, 0.0)
    meets = (apart_x <= reach_x + berth) & (apart_y <= reach_y + berth) & there
    worth, gets, costs, lost_at = 0.0, 0, 0, len(xs)
    firsts = [int(hits[0]) if hits.size else -1 for hits in (np.flatnonzero(meets[:, c]) for c in range(scene.worth.size))]
    if scene.pushes is not None and speeds is not None and scene.gentle.size:
        # Contact can begin between two imagined pictures. The greatest
        # endpoint speed bounds a constant push's speed through that step.
        through_step = np.maximum(speeds, np.r_[math.hypot(*scene.going), speeds[:-1]])
        for column, first in enumerate(firsts):
            if first >= 0 and not costly[column] and through_step[first] > scene.gentle[column]:
                costly = costly.copy()
                costly[column] = True
    for column, first in enumerate(firsts):
        if first >= 0 and costly[column]:
            costs += 1
            worth += -LOSS * caution * (SOONER ** first)
            lost_at = min(lost_at, first)
    for column, first in enumerate(firsts):
        # What is got after she has been hit is not got.
        if first >= 0 and not costly[column] and first < lost_at:
            gets += 1
            worth += scene.worth[column] * (SOONER ** first)
    return worth, gets, costs


def _barred(barred: Any, x: float, y: float) -> bool:
    if not barred:
        return False
    cells, cell = barred
    return bool(cell) and (int(x // cell), int(y // cell)) in cells
