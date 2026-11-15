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

__all__ = ["apart", "decode", "grey", "grow", "median", "middle_values", "pieces", "shrink"]

_EIGHT_WAYS = np.ones((3, 3), dtype=bool)


def decode(data: bytes) -> np.ndarray | None:
    """A PNG or JPEG screenshot as an RGB array, or None when it is not a picture."""
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(io.BytesIO(data)) as image:
            return np.asarray(image.convert("RGB"))
    # not a failure: bytes that are not a picture give none
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
    return middle_values(picture, (reach, reach))


#: The most distinct values a picture's values may take for the middle of them to be counted rather than sorted.
COUNTED_UP_TO = 4096
_COUNTING: dict[str, object] = {}


#: How the values beyond the edge are taken: the edge carried outward, or the values mirrored about it (scipy's names),
#: and numpy's names for the same.
_BEYOND = {"nearest": "edge", "reflect": "symmetric"}


def middle_values(values: np.ndarray, size: tuple[int, int], mode: str = "nearest") -> np.ndarray:
    """Each value replaced by the middle of the values in the ``size`` (down, across) around it, channel by channel:
    exactly ``scipy.ndimage.median_filter(values, size=size (+ (1,) by channel), mode=mode)``, value for value, the
    middle of an even count being the upper of the two, beyond the edge the edge carried outward ("nearest") or the
    values mirrored about it ("reflect").

    Found by counting, not by sorting. Where the values are whole numbers of a small range, as a picture's are, the
    values in a window are kept as a count of each, and moving the window one place along takes one column out of the
    count and puts one in, the middle found again from where it was (Huang, Yang and Tang, 1979): a few dozen steps a
    pixel where sorting the window took hundreds. Measured offline 2026-10-12 on a game's screen: the filter was three
    quarters of finding what looks pressable and what is drawn, read at every look. Other values are sorted as before.
    """
    from scipy import ndimage

    values = np.asarray(values)
    down, across = int(size[0]), int(size[1])
    whole = values.dtype.kind in "ui" or values.dtype == bool
    count = _counting()
    if (count is None or not whole or values.ndim not in (2, 3) or values.size == 0 or down < 1 or across < 1
            or mode not in _BEYOND or int(values.min()) < 0 or int(values.max()) >= COUNTED_UP_TO
            # Mirrored more than once over, as a window wider than the values is, the two ways of mirroring part.
            or (mode == "reflect" and (down // 2 > values.shape[0] or across // 2 > values.shape[1]))):
        full = (down, across, 1) if values.ndim == 3 else (down, across)
        return ndimage.median_filter(values, size=full, mode=mode)
    planes = values[..., None] if values.ndim == 2 else values
    out = np.empty(planes.shape, dtype=values.dtype)
    bins = int(planes.max()) + 1
    middle = (down * across) // 2
    for channel in range(planes.shape[2]):
        plane = np.ascontiguousarray(planes[..., channel], dtype=np.int32)
        padded = np.pad(plane, ((down // 2, down - down // 2 - 1), (across // 2, across - across // 2 - 1)),
                        mode=_BEYOND[mode])
        found = np.empty(plane.shape, dtype=np.int32)
        count(padded, down, across, bins, middle, found)
        out[..., channel] = found
    return out[..., 0] if values.ndim == 2 else out


def _counting() -> object | None:
    """The counting itself, compiled once (core/perception/counted_middle.py); None where it cannot be, and the values
    are sorted instead."""
    if "kernel" not in _COUNTING:
        try:
            from core.perception.counted_middle import count_the_middle
        # not a failure: where the counting cannot be compiled the values are sorted instead, as said above
        except ImportError:
            count_the_middle = None
        _COUNTING["kernel"] = count_the_middle
    return _COUNTING["kernel"]


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
