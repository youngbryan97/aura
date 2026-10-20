"""The words in a picture, read by the Mac's own text recognition, on its Neural Engine and not her model's GPU.

Bryan, 20 October 2026: her eyes and her voice should both work at once without slowing each other. MEASURED that
day while she watched YouTube: her vision model looked about every 11 seconds, a look taking 4.1 seconds (the middle
of seven), on the GPU her 27B decodes on, and her 27B fell from 16 tokens a second to 6. Both read their weights from
the same memory for every token they make; the cost adds.

What a video shows in words (its captions, its titles, a sign, a label on a part) is most of what a remark about it
can say, and reading words does not need a language model. Apple's Vision framework (VNRecognizeTextRequest) reads
them on the Neural Engine, which her models do not use: 0.45 seconds for the video in his screenshot of it.
"""
from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger("Aura.WordsOnScreen")

__all__ = ["words_in", "how_long_reading_takes"]

_TOOK: list[float] = []


def how_long_reading_takes() -> float:
    """The middle of the last readings' lengths in seconds, 0.0 before any."""
    took = sorted(_TOOK[-9:])
    return took[len(took) // 2] if took else 0.0


def words_in(picture: bytes, *, least_confidence: float = 0.4) -> list[str]:
    """The lines of words in ``picture`` (PNG or JPEG bytes), top to bottom; [] where there are none or no reader."""
    began = time.monotonic()
    try:
        import Quartz
        import Vision
        from Foundation import NSData
    except ImportError as exc:
        logger.debug("no text recognition here: %s", exc)
        return []
    data = NSData.dataWithBytes_length_(picture, len(picture))
    source = Quartz.CGImageSourceCreateWithData(data, None)
    image = Quartz.CGImageSourceCreateImageAtIndex(source, 0, None) if source is not None else None
    if image is None:
        return []
    request = Vision.VNRecognizeTextRequest.alloc().init()
    request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    request.setUsesLanguageCorrection_(True)
    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(image, {})
    ok, _error = handler.performRequests_error_([request], None)
    lines: list[tuple[float, float, str]] = []
    for observation in (request.results() or []) if ok else []:
        candidates = observation.topCandidates_(1)
        if not candidates:
            continue
        best: Any = candidates[0]
        if float(best.confidence()) < least_confidence:
            continue
        box = observation.boundingBox()  # normalized, origin at the bottom left
        lines.append((-float(box.origin.y + box.size.height), float(box.origin.x), str(best.string())))
    _TOOK.append(time.monotonic() - began)
    del _TOOK[:-20]
    return [text for _top, _left, text in sorted(lines)]
