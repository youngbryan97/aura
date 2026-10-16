"""Map a tracker's sample centres to source-image pixel centres explicitly."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from core.verify import invariant


@dataclass(frozen=True)
class SampleGrid:
    image: tuple[int, int]
    sampled: tuple[int, int]
    working: tuple[int, int]

    def __post_init__(self) -> None:
        if any(not isinstance(size, tuple) or len(size) != 2
               or any(type(v) is not int or not 1 <= v <= 4096 for v in size)
               or math.prod(size) > 8_000_000 for size in (self.image, self.sampled, self.working)):
            raise ValueError('a pixel grid needs bounded measured image, sampling and working dimensions')

    @property
    def pixel_size(self) -> float:
        return max(self.working[axis]/self.sampled[axis] for axis in (0, 1))

    @property
    def working_scale(self) -> float:
        return max(self.working[axis]/self.image[axis] for axis in (0, 1))

    def to_image(self, positions: object) -> np.ndarray:
        """The provider publishes native-index times working/native, without an offset."""
        image, sampled, working = (np.asarray(size[::-1], float) for size in (self.image, self.sampled, self.working))
        return np.asarray(positions, float)*(image/working)+(image/sampled-1)/2

    def from_image(self, positions: object) -> np.ndarray:
        image, sampled, working = (np.asarray(size[::-1], float) for size in (self.image, self.sampled, self.working))
        return (np.asarray(positions, float)-(image/sampled-1)/2)*(working/image)


@invariant('perception.point_grids_keep_sample_centres', scope='perception',
           owner='core/perception/the_pixel_grid_a_mark_uses.py', observational=False)
def _sample_grid_invariant() -> tuple:
    grid = SampleGrid((32, 32), (8, 8), (16, 16))
    assert tuple(grid.to_image((6., 8.))) == (13.5, 17.5)
    assert tuple(grid.from_image((13.5, 17.5))) == (6., 8.)
    return ()
