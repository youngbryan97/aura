"""Reading a picture keeps nothing of it once the words are out.

The picture's bytes were handed to CoreGraphics as a Python buffer with no
release callback, so each reading stayed in memory for good: 2.6 MB a reading
of the 2048 window. LIVE 2026-09-25 her eyes grew to 3.2 GB in 23 minutes of
play, until the runaway budget refused new work.
"""
from __future__ import annotations

import gc

import numpy as np
import pytest

from core.perception.what_the_pixels_show import recognize_text

psutil = pytest.importorskip("psutil")
pytest.importorskip("Vision")


def test_a_hundred_readings_do_not_keep_a_hundred_pictures():
    picture = np.full((870, 738, 3), 240, np.uint8)
    picture[300:400, 300:420] = (60, 60, 60)
    me = psutil.Process()
    recognize_text(picture)
    gc.collect()
    before = me.memory_info().rss
    for _ in range(100):
        recognize_text(picture)
    gc.collect()
    grew = (me.memory_info().rss - before) / 1e6
    # A hundred kept pictures would be about 257 MB.
    assert grew < 60, f"{grew:.0f} MB kept over 100 readings"


def test_the_bytes_go_to_data_that_owns_them():
    import inspect

    from core.perception import what_the_pixels_show

    source = inspect.getsource(what_the_pixels_show)
    assert "CGDataProviderCreateWithData(" not in source
    assert "objc.autorelease_pool()" in source
