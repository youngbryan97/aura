"""Return a current menu to navigation before experimenting with its animation.

OCR can finish after its picture has gone. A label can interrupt body trials
only when its captured word pixels still occupy the same owned surface.
Two handoffs per unchanged menu bound controls that do not lead anywhere.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

from core.agency.pixels_inside_control_trials import FrozenPixelFrame, freeze_frame
from core.language.a_way_on import asks_to_choose, confirms
from core.perception.observed_transfer import PixelSnapshot, ScreenReading, make_snapshot
from core.perception.words_kept_with_their_pixels import WordsWithPixels, words_with_their_pixels
from core.verify import invariant


@dataclass(frozen=True)
class NavigationReading:
    frame: FrozenPixelFrame
    words: WordsWithPixels
    labels: tuple[str, ...]


class WordsFromCapturedPixels(list):
    """Keep the reading's capture outside its public list of OCR regions."""

    def __init__(self, regions: list, reading: NavigationReading | None) -> None:
        super().__init__(regions)
        self.navigation_reading = reading


def capture_before_reading(picture: Any, at: Any) -> tuple[FrozenPixelFrame, PixelSnapshot] | None:
    """Freeze the actual RGB frame before an off-thread reader receives BGR."""
    frame = freeze_frame(picture, at)
    snapshot = make_snapshot(picture) if frame is not None else None
    return (frame, snapshot) if frame is not None and snapshot is not None else None


def words_bound_to_capture(regions: list, captured: tuple | None) -> WordsFromCapturedPixels:
    """Only whole navigation labels or an explicit chooser acquire menu evidence."""
    reading = None
    if captured is not None:
        frame, snapshot = captured
        bounded = [dict(region) for region in regions[:128] if isinstance(region, dict)]
        targets = [region for region in bounded if confirms(str(region.get('text') or ''))]
        if asks_to_choose(' '.join(str(region.get('text') or '') for region in bounded)):
            targets = bounded
        words = words_with_their_pixels(ScreenReading({'layout': targets}, picture_snapshot=snapshot))
        if words is not None:
            labels = tuple(sorted(str(region.get('text') or '').strip().casefold()[:80] for region in targets))
            reading = NavigationReading(frame, words, labels)
    return WordsFromCapturedPixels(regions, reading)


def keep_navigation_reading(run: Any, regions: Any) -> None:
    """A completed empty or non-menu reading retires the previous menu evidence."""
    run.navigation_reading = getattr(regions, 'navigation_reading', None)


def navigation_before_body_experiments(run: Any, hers: Any, picture: Any, at: Any,
                                      keep: dict, *, now: float) -> str:
    """Request navigation only with current pixels; supply no body or victory claim."""
    if hers.number is not None or hers.follows_pointer:
        return ''
    reading = getattr(run, 'navigation_reading', None)
    if not isinstance(reading, NavigationReading):
        return ''
    current = freeze_frame(picture, at)
    previous = reading.frame
    if (current is None or not math.isfinite(now) or not previous.valid()
            or (current.surface, current.epoch, current.source, current.width, current.height)
            != (previous.surface, previous.epoch, previous.source, previous.width, previous.height)
            or current.sequence < previous.sequence or current.lower < previous.lower
            or not current.upper <= current.arrived <= now
            or now - current.lower > .5 or reading.words.still_in(picture) is not True):
        return ''
    # A label's flashing colour cannot renew the budget. Its wording and
    # extents, on this surface, identify the unchanged navigation request.
    key = hashlib.sha256(json.dumps((current.surface, current.epoch, reading.labels,
                                    tuple(box for box, _pixels in reading.words.patches))).encode()).hexdigest()
    budgets = keep.setdefault('pixel_navigation_handoffs', {})
    if budgets.get(key, 0) >= 2:
        return ''
    budgets[key] = budgets.get(key, 0) + 1
    if len(budgets) > 64:
        del budgets[next(iter(budgets))]
    return 'a current navigation control is shown'


@invariant('agency.navigation_precedes_trials_only_on_current_word_pixels', scope='agency',
           owner='core/agency/navigation_before_body_experiments.py', observational=False)
def _current_navigation_invariant() -> tuple:
    from types import SimpleNamespace

    import numpy as np

    from core.perception.frames_as_they_are_drawn import CanvasFrameTime

    picture = np.zeros((64, 64, 3), np.uint8)
    pattern = ((np.indices((32, 32)).sum(axis=0) // 4) % 2 * 240).astype(np.uint8)
    picture[16:48, 16:48] = pattern[:, :, None]
    at = CanvasFrameTime(10., lower=10., upper=10.01, arrived=10.02,
                         source='canvas-bracket', source_at=None, epoch=1, surface='one', sequence=1)
    regions = [{'text': 'PLAY', 'x': .25, 'y': .25, 'width': .5, 'height': .5}]
    run = SimpleNamespace()
    keep_navigation_reading(run, words_bound_to_capture(regions, capture_before_reading(picture, at)))
    hers = SimpleNamespace(number=None, follows_pointer=False)
    assert navigation_before_body_experiments(run, hers, picture, at, {}, now=10.03)
    assert not navigation_before_body_experiments(run, hers, np.full_like(picture, 60), at, {}, now=10.03)
    assert not navigation_before_body_experiments(run, hers, picture, 10., {}, now=10.03)
    return ()
