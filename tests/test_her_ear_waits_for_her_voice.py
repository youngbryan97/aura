"""Her ear keeps what it hears and waits while her voice uses the GPU, then writes it all down.

MEASURED 20 October 2026: beside her 27B the ear's model took 0.42 of real time on the GPU, and while she watched her
27B decoded at 6 to 9 tokens a second where alone it decodes at 21.
"""
from __future__ import annotations

import io
import sys
import threading
import time
import types

import numpy as np

from core.senses import her_ear


def test_held_the_sound_waits_and_none_of_it_is_lost(monkeypatch, tmp_path):
    hold = tmp_path / "hold"
    hold.touch()
    fed: list[int] = []
    when: dict[str, float] = {}

    class _Stream:
        finalized_tokens: list = []

        def add_audio(self, chunk):
            when.setdefault("first_fed", time.monotonic())
            fed.append(int(chunk.shape[0]))

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

    class _Model:
        def transcribe_stream(self, context_size=None):
            return _Stream()

    monkeypatch.setitem(sys.modules, "parakeet_mlx", types.SimpleNamespace(from_pretrained=lambda _w: _Model()))
    monkeypatch.setattr(her_ear, "STEP_S", 0.01)

    def capture(_bundle, on_audio):
        def feed():
            for _ in range(5):
                on_audio(np.zeros(1600, dtype=np.float32))
            def released():
                when["released"] = time.monotonic()
                hold.unlink()
            threading.Timer(0.15, released).start()  # her voice is done
        feed()
        return object()

    monkeypatch.setattr(her_ear, "_capture", capture)
    her_ear.listen("com.apple.Safari", out=io.StringIO(), at_most_s=0.4, weights="unused", hold=str(hold))
    assert when["first_fed"] >= when["released"]  # nothing written down while held
    assert sum(fed) == 5 * 1600  # and none of what was heard lost
