"""The few things done to a picture to see what moves in it, without OpenCV.

Her primary process on macOS refuses OpenCV (core/media/safe_imports.py): its
AVFoundation stack collides with the one speech-to-text loads. LIVE 2026-10-04,
asked to mend and play a broken Pong, the repair failed on its first picture
with "OpenCV import is blocked", though every offline run had passed, because
the offline runs never installed the guard. What following things in a picture
needs from a vision library is small: decode a screenshot, shrink it, make it
grey, compare two, grow a mask by a pixel, find its connected pieces, and a
median filter once per screen. numpy, scipy and Pillow do all of it in the
same process the person is talking to.

The results have OpenCV's shapes (the stats rows of connected pieces are
x, y, w, h, area; label 0 is the background), so callers read them the same.
"""
from __future__ import annotations

import io

import numpy as np

__all__ = ["apart", "decode", "grey", "grow", "median", "pieces", "shrink"]

_EIGHT_WAYS = np.ones((3, 3), dtype=bool)


def decode(data: bytes) -> np.ndarray | None:
    """A PNG or JPEG screenshot as an RGB array, or None when it is not a picture."""
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(io.BytesIO(data)) as image:
            return np.asarray(image.convert("RGB"))
    except (UnidentifiedImageError, OSError, ValueError):
        return None


def shrink(picture: np.ndarray, wide: int, tall: int) -> np.ndarray:
    """A picture made smaller by averaging the pixels each new one covers."""
    from PIL import Image

    return np.asarray(Image.fromarray(np.ascontiguousarray(picture)).resize((wide, tall), Image.Resampling.BOX))


def grey(picture: np.ndarray) -> np.ndarray:
    """Brightness as the eye weighs the three colours (ITU-R BT.601), 0-255."""
    weighted = picture[..., 0] * 0.299 + picture[..., 1] * 0.587 + picture[..., 2] * 0.114
    return np.clip(weighted + 0.5, 0, 255).astype(np.uint8)


def apart(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """How far apart two pictures are, pixel by pixel, in the pictures' own units."""
    return np.abs(a.astype(np.int16) - b.astype(np.int16)).astype(np.uint8)


def grow(mask: np.ndarray) -> np.ndarray:
    """A mask grown by one pixel every way."""
    from scipy import ndimage

    return ndimage.binary_dilation(mask.astype(bool), structure=_EIGHT_WAYS).astype(np.uint8)


def median(picture: np.ndarray, reach: int) -> np.ndarray:
    """Each pixel replaced by the middle value around it, colour by colour."""
    from scipy import ndimage

    size = (reach, reach, 1) if picture.ndim == 3 else (reach, reach)
    return ndimage.median_filter(picture, size=size, mode="nearest")


def pieces(mask: np.ndarray) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """The connected pieces of a mask, eight ways: count, labels, stats and centres.

    As OpenCV gives them: ``count`` includes the background as label 0;
    ``stats[i]`` is x, y, w, h, area; ``centres[i]`` is x, y.
    """
    from scipy import ndimage

    labels, found = ndimage.label(mask.astype(bool), structure=_EIGHT_WAYS)
    count = found + 1
    stats = np.zeros((count, 5), dtype=np.int64)
    centres = np.zeros((count, 2), dtype=np.float64)
    if found:
        areas = np.bincount(labels.ravel(), minlength=count)
        rows, cols = np.indices(labels.shape)
        centres[:, 0] = np.bincount(labels.ravel(), weights=cols.ravel(), minlength=count) / np.maximum(areas, 1)
        centres[:, 1] = np.bincount(labels.ravel(), weights=rows.ravel(), minlength=count) / np.maximum(areas, 1)
        for index, where in enumerate(ndimage.find_objects(labels), start=1):
            if where is None:
                continue
            top, left = where[0].start, where[1].start
            stats[index] = (left, top, where[1].stop - left, where[0].stop - top, areas[index])
        stats[0] = (0, 0, labels.shape[1], labels.shape[0], areas[0])
    else:
        stats[0] = (0, 0, mask.shape[1], mask.shape[0], mask.size)
    return count, labels, stats, centres
