"""Quiet scenery leaves point capacity for newly changing regions.

Change asks for observation, never for control attribution. Every admitted
corner starts a new identity; only later matched pixels can supply motion.
"""
from __future__ import annotations

import math

import numpy as np

from core.perception.points_that_keep_their_motion import SeenPoint
from core.verify import invariant


def share_point_capacity(points: list[SeenPoint], activity: np.ndarray, scale: tuple,
                         limit: int, *, quiet_budget: int | None = None) -> tuple[SeenPoint, ...]:
    """Reserve quiet reference marks while sharing the remaining slots among changing regions."""
    from core.perception.points_before_the_picture_is_reduced import PointsBeforeReduction

    moving, exposed, quiet = [], [], []
    for point in points:
        x, y = int(round(point.x)), int(round(point.y))
        changed = 0 <= y < activity.shape[0] and 0 <= x < activity.shape[1] and activity[y, x]
        travel = math.hypot((point.x - point.old_x) * scale[0], (point.y - point.old_y) * scale[1])
        if point.pictures >= 2 and travel >= .2:
            moving.append(point)
        elif changed:
            exposed.append(point)
        else:
            quiet.append(point)
    # A changing cell shares its own slots before competing with long-lived
    # reference cells. Unused reservations return to the other observations.
    quiet_budget = limit // 3 if quiet_budget is None else min(limit, max(0, quiet_budget))
    active_budget = limit - min(quiet_budget, len(quiet))
    active = list(PointsBeforeReduction._spatial_share(moving, scale, active_budget))
    room = active_budget - len(active)
    active.extend(PointsBeforeReduction._spatial_share(exposed, scale, room) if room else ())
    kept = active + list(PointsBeforeReduction._spatial_share(quiet, scale, limit - len(active)))
    names = {p.number for p in kept}
    unused = [p for p in (*moving, *exposed, *quiet) if p.number not in names]
    kept.extend(PointsBeforeReduction._spatial_share(unused, scale, limit - len(kept)) if len(kept) < limit else ())
    return tuple(kept[:limit])


@invariant('perception.changing_regions_receive_point_capacity', scope='perception',
           owner='core/perception/points_shared_with_changing_regions.py', observational=False)
def _point_capacity_invariant() -> tuple:
    quiet = [SeenPoint(i, 10. + i, 10., 10. + i, 10., 0., 1., 1.1, .01, 20) for i in range(12)]
    fresh = [SeenPoint(i, 10. + i, 30., 10. + i, 30., 1.1, 1.1, 1.1, 0., 1) for i in range(12, 18)]
    activity = np.zeros((48, 64), bool)
    activity[25:35] = True
    kept = share_point_capacity(quiet + fresh, activity, (1., 1.), 9)
    assert len(kept) == 9 and sum(p.number >= 12 for p in kept) == 6
    assert all(p.pictures == 1 and p.before == p.at for p in kept if p.number >= 12)
    assert any(p.number < 12 for p in kept)
    return ()
