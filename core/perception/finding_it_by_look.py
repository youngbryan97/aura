"""Finding a thing on screen by how it looks, where nothing on screen names it.

Her trips go to things she can read, and most things in a world carry no
label. SIMA 2 takes a picture as an instruction — "go to the thing that looks
like this" — and its colour words ("the red house") ground in what it sees
(arXiv 2512.04797, §4.1). Here both are done without asking any model: an
example picture is found in the frame by matching features, and a colour by
the pixels that are that colour. Each answer is a sighting in the same form
her eyes report text, so a trip walks to it the same way.

An example is found by ORB features, kept only where the best match is
clearly better than the second best (Lowe's ratio test, at the ratio of his
2004 paper), and placed where a homography fitted through the kept matches by
RANSAC carries the example's corners. Fewer matches than a homography needs
means it is not there.

A colour is found without OpenCV, which Aura's main process refuses, among
the pixels coloured enough to have one: the cut
between grey and coloured is set per frame by Otsu's method on saturation,
so nothing here says how saturated is saturated. Each coloured pixel belongs
to the named hue it is nearest, and the largest joined patch of the asked
colour is the thing.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger("Aura.FindingItByLook")

__all__ = ["HUES", "looks_like", "of_colour"]

#: Named hues on OpenCV's scale, where hue runs 0 to 180.
HUES: dict[str, float] = {
    "red": 0.0, "orange": 15.0, "yellow": 30.0, "green": 60.0,
    "cyan": 90.0, "blue": 120.0, "purple": 140.0, "pink": 165.0,
}

#: Lowe's ratio: a match counts when it is this much closer than the next best.
_LOWE = 0.75

#: The fewest matches a homography can be fitted through.
_A_HOMOGRAPHY = 4


def _grey(image: Any) -> np.ndarray:
    import cv2  # noqa: PLC0415

    pixels = np.asarray(image)
    if pixels.ndim == 3:
        pixels = cv2.cvtColor(pixels.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    return pixels.astype(np.uint8)


def _a_sighting(name: str, box: tuple[float, float, float, float], high: int, wide: int, **also: Any) -> dict:
    left, top, right, bottom = box
    left, right = max(0.0, left), min(float(wide), right)
    top, bottom = max(0.0, top), min(float(high), bottom)
    return {
        "text": name,
        "x": round(left / wide, 5),
        "y": round(top / high, 5),
        "width": round((right - left) / wide, 5),
        "height": round((bottom - top) / high, 5),
        "center_x": round((left + right) / 2.0 / wide, 5),
        "center_y": round((top + bottom) / 2.0 / high, 5),
        **also,
    }


def looks_like(frame: Any, example: Any, *, name: str) -> dict | None:
    """Where in ``frame`` the thing in ``example`` is, as a sighting called ``name``. None when it is not there."""
    try:
        import cv2  # noqa: PLC0415
    except ImportError as why:
        # Aura's main process refuses OpenCV; finding by example is for a
        # process that allows it, and here it finds nothing rather than fail.
        logger.info("cannot look for %r by example here: %s", name, why)
        return None

    seen, wanted = _grey(frame), _grey(example)
    orb = cv2.ORB_create()
    # ORB leaves a patch's width unexamined at each edge and describes a
    # patch at a time, so an example only a couple of patches across has
    # almost no inside to find features in: 15 at 96 pixels, 353 at 192.
    # Smaller than four patches across, it is scaled up to four.
    across = 4 * orb.getPatchSize()
    if min(wanted.shape) < across:
        scale = across / float(min(wanted.shape))
        wanted = cv2.resize(wanted, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    kp_wanted, des_wanted = orb.detectAndCompute(wanted, None)
    kp_seen, des_seen = orb.detectAndCompute(seen, None)
    if des_wanted is None or des_seen is None or len(kp_seen) < 2:
        return None
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(des_wanted, des_seen, k=2)
    good = [one for one, *rest in pairs if rest and one.distance < _LOWE * rest[0].distance]
    if len(good) < _A_HOMOGRAPHY:
        return None
    source = np.float32([kp_wanted[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    target = np.float32([kp_seen[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    homography, inliers = cv2.findHomography(source, target, cv2.RANSAC)
    if homography is None or inliers is None or int(inliers.sum()) < _A_HOMOGRAPHY:
        return None
    high, wide = wanted.shape
    corners = np.float32([[0, 0], [wide, 0], [wide, high], [0, high]]).reshape(-1, 1, 2)
    placed = cv2.perspectiveTransform(corners, homography).reshape(-1, 2)
    box = (placed[:, 0].min(), placed[:, 1].min(), placed[:, 0].max(), placed[:, 1].max())
    return _a_sighting(name, box, seen.shape[0], seen.shape[1], matches=int(inliers.sum()))


def _hue_and_saturation(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hue on OpenCV's 0 to 180 scale and saturation on 0 to 255, from blue-green-red pixels."""
    pixels = bgr[..., :3].astype(np.float64) / 255.0
    blue, green, red = pixels[..., 0], pixels[..., 1], pixels[..., 2]
    most, least = pixels.max(axis=2), pixels.min(axis=2)
    spread = most - least
    with np.errstate(divide="ignore", invalid="ignore"):
        saturation = np.where(most > 0, spread / most, 0.0)
        hue = np.where(
            spread == 0, 0.0,
            np.where(
                most == red, ((green - blue) / spread) % 6.0,
                np.where(most == green, (blue - red) / spread + 2.0, (red - green) / spread + 4.0),
            ),
        )
    return hue * 30.0, saturation * 255.0


def _otsu(values: np.ndarray) -> float:
    """Otsu's threshold: the cut that best separates the values into two groups."""
    counts, edges = np.histogram(values, bins=256, range=(0.0, 256.0))
    share = counts / max(1, counts.sum())
    middles = (edges[:-1] + edges[1:]) / 2.0
    below = np.cumsum(share)
    mean_below = np.cumsum(share * middles)
    whole = mean_below[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (whole * below - mean_below) ** 2 / (below * (1.0 - below))
    between = np.nan_to_num(between)
    return float(middles[int(np.argmax(between))])


def of_colour(frame: Any, colour: str, *, name: str = "") -> dict | None:
    """The largest patch of a named colour in ``frame``, as a sighting. None when there is none.

    Numpy and scipy only: Aura's main process refuses OpenCV, and a trip to
    "the red door" runs there.
    """
    from scipy import ndimage  # noqa: PLC0415

    wanted = str(colour or "").strip().lower()
    if wanted not in HUES:
        return None
    pixels = np.asarray(frame)
    if pixels.ndim != 3:
        return None
    hue, saturation = _hue_and_saturation(pixels)
    coloured = saturation > _otsu(saturation)
    names = list(HUES)
    centres = np.array([HUES[one] for one in names])
    apart = np.abs(hue[..., None] - centres[None, None, :])
    apart = np.minimum(apart, 180.0 - apart)  # hue goes round
    mask = coloured & (np.argmin(apart, axis=2) == names.index(wanted))
    labels, count = ndimage.label(mask, structure=np.ones((3, 3)))
    if count == 0:
        return None
    sizes = np.bincount(labels.ravel())[1:]
    biggest = int(np.argmax(sizes)) + 1
    rows, columns = ndimage.find_objects(labels)[biggest - 1]
    return _a_sighting(
        name or wanted, (columns.start, rows.start, columns.stop, rows.stop),
        pixels.shape[0], pixels.shape[1], pixels_seen=int(sizes[biggest - 1]),
    )
