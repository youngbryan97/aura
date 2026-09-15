"""Three loop-thread stalls from one afternoon's stall dumps, each fixed at its cause.

data/error_logs/stalls on 2026-09-15 named the loop thread's frame at the
moment of each stall:

- 5.2s in ``wake_word._get_latest_transcript`` inside ``Path.resolve()`` — a
  realpath walk, five times a second, on the loop.
- 5.6s in ``liquid_substrate.encode_text_to_stimulus`` — the 512x260 random
  projection redrawn on every broadcast winner.
- 5.4s in ``earned_metric.recurrence_verdict`` — 1,300 small NumPy calls per
  verdict, each a GIL round trip against busy worker threads.
"""
from __future__ import annotations

import asyncio
import re
from pathlib import Path

import numpy as np

from core.consciousness import liquid_substrate
from core.verify import earned_metric

ROOT = Path(__file__).resolve().parent.parent


def test_the_wake_word_poll_resolves_its_sidecar_once_and_reads_it_off_the_loop():
    source = (ROOT / "core/voice/wake_word.py").read_text(encoding="utf-8")
    body = source[source.index("def _get_latest_transcript") :]
    body = body[: body.index("\n    def ", 10)]
    assert "resolve()" not in body
    assert "_AUDIO_SIDECAR" in body
    assert re.search(r"^_AUDIO_SIDECAR = Path\(__file__\)\.resolve\(\)", source, re.M)
    loop = source[source.index("async def _detection_loop") : source.index("def _get_latest_transcript")]
    assert "await asyncio.to_thread(self._get_latest_transcript)" in loop


def test_the_text_projection_is_drawn_once_per_substrate_size():
    liquid_substrate._TEXT_PROJECTIONS.clear()
    first = liquid_substrate._text_projection(64)
    second = liquid_substrate._text_projection(64)
    assert first is second
    assert first.shape == (64, 260)
    # Same draw as before the cache: seeded by the neuron count.
    expected = np.random.RandomState(64).randn(64, 260).astype(np.float32) * (1.0 / np.sqrt(260))
    np.testing.assert_array_equal(first, expected)


def test_the_vectorised_null_is_the_loop_null():
    rng = np.random.default_rng(7)
    states = rng.normal(size=(20, 32))
    unit = states / np.linalg.norm(states, axis=1)[:, None]
    gram = unit @ unit.T
    orders = np.stack([np.random.default_rng(i).permutation(20) for i in range(16)])

    by_loop = np.array(
        [
            float(np.max(earned_metric._lag_profile_from_gram(gram[np.ix_(o, o)])))
            for o in orders
        ]
    )
    np.testing.assert_allclose(earned_metric._null_maxima(gram, orders), by_loop, rtol=0, atol=1e-12)


def test_the_verdict_takes_few_array_calls(monkeypatch):
    calls = {"diagonal": 0}
    real = np.diagonal

    def counting(*args, **kwargs):
        calls["diagonal"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(earned_metric.np, "diagonal", counting)
    history = [np.random.default_rng(i).normal(size=16) for i in range(20)]
    verdict = earned_metric.recurrence_verdict(history, percentile=95.0, surrogates=128)
    assert verdict is not None
    # One observed profile (10 lags) plus one call per lag across all surrogates.
    assert calls["diagonal"] == 20
