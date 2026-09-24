"""A decode rate is timed over decoding, not over the whole stream.

The worker timed tokens from the start of the stream, which is before the
prompt is read. On the 27B, reading at 130 tokens a second and decoding at
six, a short answer's stream is mostly reading: MLX decoded 61 tokens at 6.1 a
second and the store kept 2.0. The answer clock then added the reading again
from its own rate, and a 96-token move was put at 79 seconds (live,
2026-09-23).
"""

from __future__ import annotations

import logging

import pytest

from core.brain.llm import mlx_worker


def test_mlx_decode_timing_is_used_where_it_was_given():
    performance = {"generation_tokens": 61, "decode_seconds": 10.0, "prefill_seconds": 20.0}
    tokens, seconds = mlx_worker._what_decoding_took(performance, 61, 30.2, 20.1)
    assert (tokens, seconds) == (61, 10.0)


def test_without_mlx_timing_the_stream_after_its_first_token_is_decoding():
    tokens, seconds = mlx_worker._what_decoding_took({}, 61, 30.0, 20.0)
    assert tokens == 60
    assert seconds == pytest.approx(10.0)


def test_with_no_clock_for_decoding_nothing_is_recorded():
    assert mlx_worker._what_decoding_took({}, 61, 30.0, None) == (0, 0.0)
    assert mlx_worker._what_decoding_took({}, 1, 30.0, 29.0) == (0, 0.0)


def test_the_worker_records_the_decoding_rate_and_not_the_stream_rate(monkeypatch):
    recorded: list[tuple[int, float, str]] = []
    monkeypatch.setattr(
        mlx_worker,
        "_record_decode_rate",
        lambda tokens, seconds, model="": recorded.append((tokens, seconds, model)),
    )
    monkeypatch.setattr(mlx_worker, "_record_read_rate", lambda chars, seconds: None)
    state: dict = {}
    started = mlx_worker.time.perf_counter() - 30.0
    mlx_worker._mlx_worker_loop_total_generated_tokens(
        3000,
        {"generation_tokens": 61, "decode_seconds": 10.0, "prefill_seconds": 20.0},
        started,
        logging.getLogger("test"),
        "/models/a-27B",
        state,
        61,
        20.1,
    )
    assert recorded == [(61, 10.0, "a-27B")]
    assert state["decode_tokens_per_second"] == pytest.approx(6.1)
