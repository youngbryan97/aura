"""A worker reading a long prompt has not stopped; it has not started.

The livelock ceiling asks how long a worker may go without producing a token
before it is wedged, and it was answered without reference to how much there
was to read. Measured live 2026-08-26: an 8,618-character prompt takes about
eighteen seconds to prefill at this host's measured rate, the ceiling was
twenty, and "Cortex still sending heartbeats (1.8s ago) but produced no token
in 20.2s. Recycling the lane." Every large question destroyed a warm 20GB
model, which made the next one slower still.
"""

from __future__ import annotations

import inspect

import pytest

from core.brain.llm import mlx_client


@pytest.fixture(autouse=True)
def measured(monkeypatch):
    monkeypatch.setitem(mlx_client._HOST_RATES, "prefill", 145.0)


class _Reading:
    """Just enough of a client to ask it what reading costs."""

    _prefill_tokens_per_s = 145.0
    _current_prompt_chars = 0

    _measured_prefill_rate = mlx_client.MLXLocalClient._measured_prefill_rate
    _prefill_floor_seconds = mlx_client.MLXLocalClient._prefill_floor_seconds


def test_a_long_prompt_costs_more_than_the_flat_ceiling():
    reading = _Reading()
    reading._current_prompt_chars = 8618
    assert reading._prefill_floor_seconds(reading._current_prompt_chars) > 20.0


def test_a_short_prompt_costs_less_and_does_not_raise_it():
    reading = _Reading()
    assert reading._prefill_floor_seconds(2000) < 20.0


def test_nothing_to_read_costs_nothing():
    assert _Reading()._prefill_floor_seconds(0) == 0.0


def test_the_livelock_ceiling_is_raised_by_what_the_reading_costs():
    source = inspect.getsource(mlx_client)
    where = source.index("livelock_ceiling = max(")
    window = source[where : where + 220]
    assert "self._prefill_floor_seconds(self._current_prompt_chars)" in window


def test_it_only_ever_raises_the_ceiling():
    """max(), so a short prompt leaves the flat ceiling exactly where it was."""
    source = inspect.getsource(mlx_client)
    where = source.index("livelock_ceiling = max(")
    assert "livelock_ceiling," in source[where : where + 220]
