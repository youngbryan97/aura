"""A model lane on the CPU is given the time a slow lane gets, not a light GPU lane's 20 seconds.

Her emergency tier is a 1.5B model on the CPU, registered with a 120 s timeout because "CPU is slow". On her day run of
3 November every request it got (a 3,355-character narrative prompt, 50 tokens) was killed at 20.0 s with no token: the
light lanes' ceiling, set for Metal. No generation ever finished, so the worker never measured its own rate, and the
reflex circuit opened 16 times.
"""
from __future__ import annotations

import pytest

from core.brain.llm.mlx_client import MLXLocalClient

pytestmark = pytest.mark.unit


def _client(device: str) -> MLXLocalClient:
    client = MLXLocalClient.__new__(MLXLocalClient)
    client.model_path = "/models/Qwen2.5-1.5B-Instruct-4bit"
    client.device = device
    client._tokens_since_spawn = 0
    return client


def test_a_cpu_lane_has_a_slow_lanes_ceiling(monkeypatch):
    monkeypatch.delenv("AURA_FIRST_TOKEN_ABSOLUTE_CEILING_S", raising=False)
    cpu, gpu = _client("cpu"), _client("gpu")
    for client in (cpu, gpu):
        monkeypatch.setattr(client, "_pressure_adaptive_stretch", lambda: (1.0, {}), raising=False)
        monkeypatch.setattr(client, "_is_deep_solver_lane", lambda: False, raising=False)
    assert gpu._first_token_absolute_ceiling() == pytest.approx(20.0)
    assert cpu._first_token_absolute_ceiling() >= 90.0
    assert cpu._first_token_absolute_ceiling(foreground_request=True) >= 120.0


def test_its_first_generation_is_not_cut_off_at_the_light_lanes_twenty_seconds(monkeypatch):
    monkeypatch.delenv("AURA_FIRST_TOKEN_ABSOLUTE_CEILING_S", raising=False)
    cpu = _client("cpu")
    monkeypatch.setattr(cpu, "_pressure_adaptive_stretch", lambda: (1.0, {}), raising=False)
    monkeypatch.setattr(cpu, "_is_deep_solver_lane", lambda: False, raising=False)
    assert cpu._cold_lane_first_token_allowance() > 20.0
