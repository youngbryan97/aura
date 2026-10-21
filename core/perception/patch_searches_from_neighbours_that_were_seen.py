"""Use independently measured neighbouring motion to propose another pixel search.

Neighbours need not belong to one object. Their fitted transform only starts
a new search. Current bidirectional colour fits and local ambiguity checks
still decide whether a missing mark can continue.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from core.perception.patches_compared_on_a_common_grid import RECOVERY_LIMIT, distinct_patches, verified_patch_journeys
from core.perception.points_that_keep_their_motion import SeenPoint, _follow
from core.perception.position_from_the_same_points import PointPose, _reference_error
from core.perception.rigid_motion_from_named_points import rigid_motion
from core.verify import invariant


@dataclass(frozen=True)
class NeighbourPatchSearch:
    number: int
    born: float
    before: float
    at: float
    position: tuple[float, float]
    angle: float
    error: float
    neighbours: tuple[tuple[int, float], ...]


def _point_values(point: SeenPoint) -> bool:
    return (type(point.number) is int and point.number >= 0 and type(point.pictures) is int and point.pictures >= 1
        and all(type(v) in (int, float) and math.isfinite(v) for v in
                (point.x, point.y, point.old_x, point.old_y, point.born, point.before, point.at, point.error))
        and 0 <= point.born <= point.before <= point.at and point.error >= 0
        and max(abs(v) for v in (point.x, point.y, point.old_x, point.old_y)) <= 8192.)


def neighbour_patch_searches(previous: tuple, measured: list, missing: tuple, before: float, at: float) -> tuple:
    """Propose at most three local rigid searches for each of 32 missing marks."""
    if (not isinstance(previous, tuple) or not isinstance(measured, list) or not isinstance(missing, tuple)
            or not 1 <= len(previous) <= 384 or len(measured) > 384 or len(missing) > RECOVERY_LIMIT
            or any(not isinstance(p, SeenPoint) for p in (*previous, *measured, *missing))
            or any(not _point_values(p) for p in (*previous, *measured, *missing))
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in (before, at))
            or not 0 <= before < at <= before+.35):
        return ()
    old = {(p.number, p.born): p for p in previous}
    if (len(old) != len(previous) or len({p.number for p in missing}) != len(missing)
            or any(p.at != before for p in previous)
            or any((p.number, p.born) not in old or old[p.number, p.born] != p for p in missing)):
        return ()
    known = [p for p in measured if (p.number, p.born) in old and p.before == before and p.at == at
             and p.pictures == old[p.number, p.born].pictures+1 and p.error <= .025
             and (p.old_x, p.old_y) == (old[p.number, p.born].x, old[p.number, p.born].y)
             and all(math.isfinite(v) for v in (p.x, p.y, p.old_x, p.old_y, p.error))]
    if len(known) < 3 or len({p.number for p in known}) != len(known):
        return ()
    result, fits = [], {}
    for point in missing:
        if any(p.number == point.number for p in known):
            continue
        nearby = sorted(known, key=lambda p: (p.old_x-point.x)**2+(p.old_y-point.y)**2)[:12]
        for radius in (14., 28., 56.):
            cohort = sorted((p for p in nearby if math.hypot(p.old_x-point.x, p.old_y-point.y) <= radius),
                            key=lambda p: p.number)
            if len(cohort) < 3:
                continue
            key = tuple((p.number, p.born) for p in cohort)
            if key not in fits:
                starts, ends = np.asarray([(p.old_x, p.old_y) for p in cohort]), np.asarray([(p.x, p.y) for p in cohort])
                fits[key] = starts, ends, rigid_motion(starts, ends, error=.5)
            starts, ends, fitted = fits[key]
            if fitted is None:
                continue
            pose = PointPose(point.born, before, (point.x, point.y), {}, (point.x, point.y), pixel_size=1.)
            error = _reference_error(pose, starts, ends, fitted, 1.)
            if not math.isfinite(error) or error > 1.:
                continue
            position = tuple(float(v) for v in fitted.apply((point.x, point.y)))
            if any(h.number == point.number and math.dist(h.position, position) < .2
                   and abs(h.angle-fitted.angle) < .02 for h in result):
                continue
            result.append(NeighbourPatchSearch(point.number, point.born, before, at, position, fitted.angle,
                error, tuple((cohort[i].number, cohort[i].born) for i in fitted.inliers)))
    return tuple(result)


def recover_from_neighbour_searches(tracker: object, image: list, colours: np.ndarray, at: float,
                                   measured: list, retained: list) -> list:
    """Newly recovered points cannot provide their own neighbourhood witnesses."""
    visible = {p.number: p for p in retained}
    missing = [p for p in tracker._native_points if p.number not in visible]
    missing.sort(key=lambda p: (tracker.still_needed.requests.get((p.number, p.born), -1.) <= at,
                                -p.pictures, p.error, p.number))
    proposals = neighbour_patch_searches(tracker._native_points, measured, tuple(missing[:RECOVERY_LIMIT]), tracker._at, at)
    if not proposals:
        return retained
    old = {p.number: p for p in tracker._native_points}
    starts = np.asarray([(old[h.number].x, old[h.number].y) for h in proposals])
    guesses = np.asarray([h.position for h in proposals])
    _coarse, forwards = _follow(tracker._image, image, starts)
    ends, error, angle, valid = verified_patch_journeys(tracker._colours, colours, tracker._image, image,
                                                       starts, guesses, forwards)
    if not len(ends):
        return retained
    valid &= (np.linalg.norm(ends-guesses, axis=1) < .5) & (abs(angle-np.asarray([h.angle for h in proposals])) < .08)
    accepted = {}
    for index, hint in enumerate(proposals):
        if valid[index]:
            accepted.setdefault(hint.number, []).append(index)
    candidates = [number for number, indices in accepted.items()
                  if max(np.linalg.norm(ends[i]-ends[indices[0]]) for i in indices) < .5]
    if candidates:
        distinctive = distinct_patches(tracker._colours, np.asarray([(old[n].x, old[n].y) for n in candidates]))
        for number, distinct in zip(candidates, distinctive, strict=True):
            if distinct:
                index = min(accepted[number], key=lambda i: error[i])
                point = old[number]
                visible[number] = SeenPoint(number, float(ends[index, 0]), float(ends[index, 1]), point.x, point.y,
                    point.born, tracker._at, at, float(error[index]), point.pictures+1)
    return [visible[p.number] for p in tracker._native_points if p.number in visible]


@invariant('perception.neighbour_motion_can_only_propose_a_patch_search', scope='perception',
           owner='core/perception/patch_searches_from_neighbours_that_were_seen.py', observational=False)
def _neighbour_patch_search_invariant() -> tuple:
    old = tuple(SeenPoint(n, x, y, x, y, 1., 1., 1., 0., 1)
                for n, (x, y) in enumerate(((20., 20.), (32., 20.), (20., 32.), (26., 26.))))
    assert neighbour_patch_searches(old, [], old[-1:], 1., 1.1) == ()
    current = [SeenPoint(p.number, p.x+3., p.y-2., p.x, p.y, p.born, 1., 1.1, .001, 2) for p in old[:3]]
    hints = neighbour_patch_searches(old, current, old[-1:], 1., 1.1)
    assert hints and hints[0].position == (29., 24.) and not isinstance(hints[0], SeenPoint)
    return ()
