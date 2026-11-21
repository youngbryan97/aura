"""A window's pixels are read through a view of Core Graphics' data, never by making bytes of it.

LIVE 21 November 2026 her process grew 2.6 GB over a seven-minute game against Stockfish, at a steady 370 MB a minute,
and her allostasis forecast her red line from it. A probe found the cause in core/capabilities/window_server.capture:
bytes() of the data PyObjC hands back kept every copy it made, 5.83 MB a look at a 1649 by 918 window, where a view of
the same data kept nothing.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import core.capabilities.window_server as window_server

WIDE, TALL = 5, 3


class _TheData:
    """Core Graphics' copy of a picture's bytes, as PyObjC hands it over: readable as a buffer."""

    def __init__(self, raw: bytes) -> None:
        self._raw = raw

    def __buffer__(self, flags: int) -> memoryview:
        return memoryview(self._raw)

    def __bytes__(self) -> bytes:
        raise AssertionError("bytes() of the data keeps every copy it makes")


def _quartz(pixels: np.ndarray) -> SimpleNamespace:
    data = _TheData(pixels.tobytes())
    return SimpleNamespace(
        CGRectNull=None, kCGWindowListOptionIncludingWindow=1, kCGWindowImageBoundsIgnoreFraming=1,
        kCGWindowImageNominalResolution=2, CGWindowListCreateImage=lambda *_a: "image",
        CGImageGetWidth=lambda _i: WIDE, CGImageGetHeight=lambda _i: TALL, CGImageGetBytesPerRow=lambda _i: WIDE * 4,
        CGImageGetDataProvider=lambda _i: "provider", CGDataProviderCopyData=lambda _p: data,
        CGImageGetBitmapInfo=lambda _i: 8192, kCGBitmapByteOrderMask=28672, kCGBitmapByteOrder32Little=8192,
        CGImageGetAlphaInfo=lambda _i: 2, kCGImageAlphaPremultipliedFirst=2, kCGImageAlphaFirst=4,
        kCGImageAlphaNoneSkipFirst=6)


def test_the_pixels_come_through_a_view_and_are_her_own(monkeypatch) -> None:
    bgra = np.arange(TALL * WIDE * 4, dtype=np.uint8).reshape(TALL, WIDE, 4)
    monkeypatch.setattr(window_server, "_quartz", lambda: _quartz(bgra))
    picture = window_server.capture(SimpleNamespace(number=7))
    assert picture is not None and picture.shape == (TALL, WIDE, 3)
    assert np.array_equal(picture, bgra[:, :, :3])
    # Her own copy: nothing holds Core Graphics' data once the picture is read.
    assert picture.flags.c_contiguous and (picture.base is None or not isinstance(picture.base, memoryview))
