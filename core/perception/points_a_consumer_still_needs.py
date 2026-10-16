"""Retain visible measured marks for bounded active consumers within the point budget."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

from core.verify import invariant


@dataclass(frozen=True)
class MaterialRegion:
    """A bounded sampling proposal, without object or control authority."""

    box: tuple[float, float, float, float]
    names: tuple[tuple[int, float], ...]

    def __post_init__(self) -> None:
        if (not isinstance(self.box, tuple) or len(self.box) != 4
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in self.box)
                or self.box[2] <= self.box[0] or self.box[3] <= self.box[1]
                or not isinstance(self.names, tuple) or not 3 <= len(self.names) <= 64
                or any(not isinstance(key, tuple) or len(key) != 2 or type(key[0]) is not int or key[0] < 0
                       or type(key[1]) not in (int, float) or not math.isfinite(key[1]) or key[1] < 0 for key in self.names)
                or len(set(self.names)) != len(self.names)):
            raise ValueError('a sampling region needs finite extents and current supporting marks')


@dataclass(frozen=True)
class MaterialRetention:
    epoch: str
    at: float
    until: float
    names: tuple[tuple[int, float], ...]
    regions: tuple[MaterialRegion, ...] = ()

    def __post_init__(self) -> None:
        if (not isinstance(self.epoch, str) or not re.fullmatch('[0-9a-f]{32}', self.epoch)
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in (self.at, self.until))
                or not 0 <= self.at < self.until <= self.at+1.
                or not isinstance(self.names, tuple) or not 1 <= len(self.names) <= 64
                or any(not isinstance(key, tuple) or len(key) != 2 or type(key[0]) is not int or key[0] < 0
                       or type(key[1]) not in (int, float) or not math.isfinite(key[1]) or not 0 <= key[1] <= self.at
                       for key in self.names) or len(set(self.names)) != len(self.names)):
            raise ValueError('a material retention request needs bounded current point identities and expiry')
        if (not isinstance(self.regions, tuple) or len(self.regions) > 4
                or any(not isinstance(region, MaterialRegion) or not set(region.names) <= set(self.names)
                       for region in self.regions)):
            raise ValueError('sampling regions need bounded retained current mark support')


class PointsStillNeeded:
    def __init__(self) -> None:
        self.requests: dict[tuple, float] = {}
        self.regions: tuple = ()

    def request(self, request: MaterialRetention, marks: object) -> int:
        from core.perception.named_material_in_a_consumed_frame import MaterialMarks

        if (not isinstance(request, MaterialRetention) or not isinstance(marks, MaterialMarks)
                or request.epoch != marks.epoch or not request.at <= marks.pixels.at < request.until):
            return 0
        current = {(p.number, p.born) for p in marks.points if p.pictures >= 2}
        for key in request.names:
            if key in current:
                self.requests[key] = max(self.requests.get(key, -math.inf), request.until)
        self.requests = dict(sorted(((key, expiry) for key, expiry in self.requests.items()
                                     if expiry > marks.pixels.at and key in current), key=lambda item: (-item[1], item[0]))[:64])
        if request.at == marks.pixels.at:
            places = {(p.number, p.born): (p.x, p.y) for p in marks.points if p.pictures >= 2}
            self.regions = tuple((region.box, tuple((key, places[key]) for key in region.names if key in places),
                                  request.until) for region in request.regions)
        return sum(key in self.requests for key in request.names)

    def priority(self, measured: list, scale: tuple, shape: tuple, at: float):
        """Advance sampling proposals only through current visible rigid support."""
        if not self.regions:
            return None
        import numpy as np
        from core.perception.rigid_motion_from_named_points import rigid_motion

        current = {(p.number, p.born): (p.x*scale[0], p.y*scale[1]) for p in measured if p.pictures >= 2}
        mask = np.zeros(shape, bool)
        self.regions = tuple(region for region in self.regions if region[2] > at)
        for box, anchors, _until in self.regions:
            retained = [(old, current[key]) for key, old in anchors if key in current]
            if len(retained) < 3:
                continue
            fit = rigid_motion(np.asarray([old for old, _new in retained]), np.asarray([new for _old, new in retained]))
            if fit is None:
                continue
            x0, y0, x1, y1 = box
            corners = fit.apply(np.asarray(((x0, y0), (x1, y0), (x0, y1), (x1, y1))))/np.asarray(scale)
            left, top = np.floor(corners.min(axis=0)).astype(int)
            right, bottom = np.ceil(corners.max(axis=0)).astype(int)
            left, right, top, bottom = max(0, left), min(shape[1], right), max(0, top), min(shape[0], bottom)
            if left < right and top < bottom and (right-left)*(bottom-top) <= mask.size//4:
                mask[top:bottom, left:right] = True
        return mask if 0 < np.count_nonzero(mask) <= mask.size//2 else None

    def keep(self, measured: list, proposed: tuple, at: float, limit: int) -> tuple:
        """A request changes capacity allocation; failed or hidden points remain absent."""
        if not self.requests:
            return proposed
        self.requests = {key: expiry for key, expiry in self.requests.items() if expiry > at}
        wanted = [p for p in measured if self.requests.get((p.number, p.born), -math.inf) > at and p.pictures >= 2]
        wanted.sort(key=lambda p: (-self.requests[(p.number, p.born)], -p.pictures, p.error, p.number))
        kept = wanted[:min(64, limit//3)]
        names = {p.number for p in kept}
        kept.extend(p for p in proposed if p.number not in names)
        return tuple(kept[:limit])


@invariant('perception.point_requests_cannot_restore_missing_material', scope='perception',
           owner='core/perception/points_a_consumer_still_needs.py', observational=False)
def _point_retention_visibility_invariant() -> tuple:
    needed = PointsStillNeeded()
    needed.requests[(7, 1.)] = 2.
    assert needed.keep([], (), 1.5, 12) == ()
    needed.keep([], (), 2., 12)
    assert not needed.requests
    return ()
