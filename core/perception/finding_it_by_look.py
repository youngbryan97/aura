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

A colour is found among the pixels coloured enough to have one: the cut
between grey and coloured is set per frame by Otsu's method on saturation,
so nothing here says how saturated is saturated. Each coloured pixel belongs
to the named hue it is nearest, and the largest joined patch of the asked
colour is the thing.
"""

from __future__ import annotations

from typing import Any

import numpy as np

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
    import cv2  # noqa: PLC0415

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


def of_colour(frame: Any, colour: str, *, name: str = "") -> dict | None:
    """The largest patch of a named colour in ``frame``, as a sighting. None when there is none."""
    import cv2  # noqa: PLC0415

    wanted = str(colour or "").strip().lower()
    if wanted not in HUES:
        return None
    pixels = np.asarray(frame).astype(np.uint8)
    if pixels.ndim != 3:
        return None
    hsv = cv2.cvtColor(pixels, cv2.COLOR_BGR2HSV)
    hue, saturation = hsv[..., 0].astype(float), hsv[..., 1]
    cut, _ = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    coloured = saturation > cut
    names = list(HUES)
    centres = np.array([HUES[one] for one in names])
    apart = np.abs(hue[..., None] - centres[None, None, :])
    apart = np.minimum(apart, 180.0 - apart)  # hue goes round
    nearest = np.argmin(apart, axis=2)
    mask = (coloured & (nearest == names.index(wanted))).astype(np.uint8)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return None
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    left, top, wide, high = (int(v) for v in stats[biggest, :4])
    return _a_sighting(
        name or wanted, (left, top, left + wide, top + high), pixels.shape[0], pixels.shape[1],
        pixels_seen=int(stats[biggest, cv2.CC_STAT_AREA]),
    )
