"""Publish measured pixel marks without requiring an object outline first."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

from core.perception.pixels_consumed_by_the_observer import PixelsConsumed
from core.perception.points_that_keep_their_motion import FollowingPoints, SeenPoint
from core.perception.the_pixel_grid_a_mark_uses import SampleGrid
from core.verify import invariant


@dataclass(frozen=True)
class MaterialMarks:
    """One bounded visible point stream in the observer's working coordinates."""

    epoch: str
    sequence: int
    pixels: PixelsConsumed
    scale: float
    pixel_size: float
    points: tuple[SeenPoint, ...]
    grid: SampleGrid | None = None

    def __post_init__(self) -> None:
        if (not isinstance(self.epoch, str) or not re.fullmatch('[0-9a-f]{32}', self.epoch)
                or type(self.sequence) is not int or not 1 <= self.sequence < 2**63
                or not isinstance(self.pixels, PixelsConsumed)
                or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0
                       for v in (self.scale, self.pixel_size))
                or self.scale > 8 or self.pixel_size > 32
                or not isinstance(self.points, tuple) or len(self.points) > 384):
            raise ValueError('material marks need a bounded current capture and sampling frame')
        if self.grid is None and self.scale == self.pixel_size == 1.:
            object.__setattr__(self, 'grid', SampleGrid(self.pixels.shape[:2], self.pixels.shape[:2], self.pixels.shape[:2]))
        if (not isinstance(self.grid, SampleGrid) or self.grid.image != self.pixels.shape[:2]
                or self.grid.working[1]/self.grid.image[1] != self.scale or self.grid.pixel_size != self.pixel_size):
            raise ValueError('material marks need their actual sampling grid')
        names = set()
        for point in self.points:
            if (not isinstance(point, SeenPoint) or type(point.number) is not int or point.number < 0
                    or point.number in names or type(point.pictures) is not int or point.pictures < 1
                    or any(type(v) not in (int, float) or not math.isfinite(v) for v in
                           (point.x, point.y, point.old_x, point.old_y, point.born, point.before, point.at, point.error))
                    or not 0 <= point.born <= point.before <= point.at or point.at != self.pixels.at
                    or point.error < 0
                    or not 0 <= point.x < self.grid.working[1]
                    or not 0 <= point.y < self.grid.working[0]):
                raise ValueError('material marks need unique visible measured point identities')
            names.add(point.number)


def material_marks_from(observer: object) -> MaterialMarks | None:
    """Read only the worker-owned tracker after it consumed this capture."""
    provider = getattr(observer, 'points', None)
    pixels = getattr(observer, 'pixels_consumed', None)
    if not isinstance(provider, FollowingPoints) or not isinstance(pixels, PixelsConsumed):
        return None
    geometry = getattr(provider, '_geometry', None)
    if geometry is None:
        if type(provider) is not FollowingPoints:
            return None
        geometry = pixels.shape[:2], getattr(observer, 'shape', ()), getattr(observer, 'shape', ())
    else:
        if (not isinstance(geometry, tuple) or len(geometry) != 3
                or any(not isinstance(size, tuple) or len(size) != 2
                       or any(type(n) is not int or not 1 <= n <= 4096 for n in size) for size in geometry)
                or geometry[0] != pixels.shape[:2] or geometry[2] != getattr(observer, 'shape', None)):
            return None
    try:
        grid = SampleGrid(*geometry)
        return MaterialMarks(getattr(observer, '_material_mark_epoch', ''), getattr(observer, 'pictures', -1)+1,
                             pixels, getattr(observer, 'scale', None), grid.pixel_size, tuple(provider.points), grid)
    except (ValueError, TypeError, AttributeError):
        return None


@invariant('perception.named_material_requires_its_consumed_capture', scope='perception',
           owner='core/perception/named_material_in_a_consumed_frame.py', observational=False)
def _named_material_capture_invariant() -> tuple:
    pixels = PixelsConsumed(1., (32, 32, 3), '0'*64)
    point = SeenPoint(7, 10., 12., 10., 12., 1., 1., 1., 0., 1)
    assert MaterialMarks('0'*32, 1, pixels, 1., 1., (point,)).points == (point,)
    try:
        MaterialMarks('0'*32, 1, PixelsConsumed(2., (32, 32, 3), '0'*64), 1., 1., (point,))
    except ValueError:
        return ()
    raise AssertionError('a point from another capture was published as current')
