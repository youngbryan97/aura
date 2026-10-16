"""A small receipt for the exact RGB array consumed by a play observer."""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass

import numpy as np

from core.verify import invariant


@dataclass(frozen=True)
class PixelsConsumed:
    at: float
    shape: tuple[int, int, int]
    sha256: str

    def __post_init__(self) -> None:
        if (type(self.at) not in (int, float) or not math.isfinite(self.at)
                or not isinstance(self.shape, tuple) or len(self.shape) != 3 or self.shape[2] != 3
                or any(type(v) is not int or not 1 <= v <= 4096 for v in self.shape)
                or math.prod(self.shape) > 24_000_000
                or not isinstance(self.sha256, str) or not re.fullmatch('[0-9a-f]{64}', self.sha256)):
            raise ValueError('a consumed-picture receipt needs bounded RGB geometry, capture time and pixel hash')


def consumed_pixels(picture: np.ndarray, at: float) -> PixelsConsumed | None:
    """Record bytes only; unsupported input cannot provide pixel custody."""
    if (type(at) not in (int, float) or not math.isfinite(at) or not isinstance(picture, np.ndarray)
            or picture.dtype != np.uint8 or picture.ndim != 3 or picture.shape[2] != 3
            or min(picture.shape[:2]) < 1 or max(picture.shape[:2]) > 4096 or picture.size > 24_000_000):
        return None
    digest = hashlib.sha256(str(picture.shape).encode()+picture.tobytes()).hexdigest()
    return PixelsConsumed(at, picture.shape, digest)


@invariant('perception.observers_keep_their_consumed_pixel_receipt', scope='perception',
           owner='core/perception/pixels_consumed_by_the_observer.py', observational=False)
def _consumed_pixel_invariant() -> tuple:
    picture = np.zeros((32, 48, 3), np.uint8)
    first = consumed_pixels(picture, 1.)
    picture[0, 0] = 1
    assert first is not None and first != consumed_pixels(picture, 1.)
    assert first.at == 1. and first.shape == (32, 48, 3)
    assert consumed_pixels(picture.astype(float), 1.) is None
    return ()
