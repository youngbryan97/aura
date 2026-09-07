"""The prefill rate a deadline is built from is the one that measured reading.

Two things in this runtime produce a number called "the prefill rate". MLX
times the prefill inside the worker. The client times the gaps between prefill
PROGRESS MESSAGES, which cross an IPC queue and land on a busy event loop, so
what it measures is how often the parent got told.

They are not estimates of the same quantity, and the slower one drove every
user-facing deadline.

LIVE, 2026-09-07, one turn: the worker logged 410-990 tok/s and the client had
learned 56. The answer clock then said a 52,020-character prompt would take 698
seconds to read — against about 30 in fact — and sized the turn against it.
"""

from __future__ import annotations

import pytest

from core.brain.llm import mlx_client as mc


class _Client:
    """Only the two methods under test, over the real implementations."""

    _measured_prefill_rate = mc.MLXLocalClient._measured_prefill_rate
    least_time_to_read = mc.MLXLocalClient.least_time_to_read
    _prefill_floor_seconds = mc.MLXLocalClient._prefill_floor_seconds


def test_an_unmeasured_worker_gets_the_pessimistic_rate() -> None:
    client = _Client()
    assert client._measured_prefill_rate() == mc._UNMEASURED_PREFILL_RATE


def test_the_progress_estimate_is_used_until_a_generation_finishes() -> None:
    client = _Client()
    client._prefill_tokens_per_s = 56.0
    assert client._measured_prefill_rate() == pytest.approx(56.0)


def test_the_worker_measurement_wins_once_there_is_one() -> None:
    client = _Client()
    client._prefill_tokens_per_s = 56.0
    client._worker_measured_prefill_tps = 556.6
    assert client._measured_prefill_rate() == pytest.approx(556.6)


def test_the_live_prompt_stops_taking_eleven_minutes_to_read() -> None:
    """The turn that produced the number, with both rates."""

    chars = 52020
    from_progress = _Client()
    from_progress._prefill_tokens_per_s = 56.0
    from_worker = _Client()
    from_worker._prefill_tokens_per_s = 56.0
    from_worker._worker_measured_prefill_tps = 556.6

    assert from_progress.least_time_to_read(chars) > 600.0
    measured = from_worker.least_time_to_read(chars)
    assert measured < 120.0
    # Still carries the headroom the floor is deliberately padded with.
    assert measured > chars / mc._CHARS_PER_TOKEN / 556.6


def test_an_unusable_report_leaves_the_rate_alone() -> None:
    """A rate that will not parse is not a rate."""

    client = _Client()
    client._worker_measured_prefill_tps = 500.0
    for bad in (0.0, -1.0, float("inf"), float("nan")):
        held = client._worker_measured_prefill_tps
        if not (bad > 0.0) or bad != bad or bad == float("inf"):
            assert client._measured_prefill_rate() == pytest.approx(held)
