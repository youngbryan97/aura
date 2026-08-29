"""A worker that has never spoken is coming up, and killing it loses the measurement.

LIVE 2026-08-29: a 1.5B was spawned, given its first job, and abandoned at the
8-second first-token SLA — "job seq=1", "abandoned_first_token_sla", "no
soft-cancel acknowledgement, worker presumed wedged; rebooting", "Circuit OPEN
for Reflex". The replacement was cold for the same reason.

The first allowance was derived from the weights, which assumes the time is
spent reading them: 0.8GB worked out to 3.2 seconds and the lane took longer
than 8 to speak, because the rest is framework import, tokenizer and shader
compile. Guessing low is what made the measurement impossible — the first
generation of a worker's life is the one that would have measured it.
"""

from __future__ import annotations

import pytest

from core.brain.llm import mlx_client

pytestmark = pytest.mark.unit


class _Lane:
    """Only the parts of the client the allowance reads."""

    model_path = "/models/Qwen2.5-1.5B-Instruct-4bit"

    _weight_gigabytes = mlx_client.MLXLocalClient._weight_gigabytes
    _cold_lane_first_token_allowance = (
        mlx_client.MLXLocalClient._cold_lane_first_token_allowance
    )

    def __init__(self, *, tokens_since_spawn: int, ceiling: float = 20.0) -> None:
        self._tokens_since_spawn = tokens_since_spawn
        self._ceiling = ceiling

    def _first_token_absolute_ceiling(self, **_kwargs: object) -> float:
        return self._ceiling


@pytest.fixture(autouse=True)
def _clean_measurements():
    cold = dict(mlx_client._COLD_FIRST_TOKEN_S)
    sizes = dict(mlx_client._WEIGHT_SIZES)
    mlx_client._COLD_FIRST_TOKEN_S.clear()
    yield
    mlx_client._COLD_FIRST_TOKEN_S.clear()
    mlx_client._COLD_FIRST_TOKEN_S.update(cold)
    mlx_client._WEIGHT_SIZES.clear()
    mlx_client._WEIGHT_SIZES.update(sizes)


def test_an_unmeasured_cold_start_gets_the_bound_that_already_exists() -> None:
    """Not an arithmetic guess: the lane's own answer to how long it may be silent."""

    assert _Lane(tokens_since_spawn=0, ceiling=20.0)._cold_lane_first_token_allowance() == 20.0


def test_a_warm_lane_gets_none_of_it() -> None:
    """The allowance covers the state, not the lane."""

    assert _Lane(tokens_since_spawn=1)._cold_lane_first_token_allowance() == 0.0


def test_once_measured_the_measurement_answers() -> None:
    mlx_client._COLD_FIRST_TOKEN_S["Qwen2.5-1.5B-Instruct-4bit"] = 4.0
    lane = _Lane(tokens_since_spawn=0, ceiling=60.0)
    assert lane._cold_lane_first_token_allowance() == 4.0 * mlx_client._COLD_START_HEADROOM


def test_the_ceiling_caps_the_measurement() -> None:
    """A measurement plus headroom must not outrun the lane's own bound."""

    mlx_client._COLD_FIRST_TOKEN_S["Qwen2.5-1.5B-Instruct-4bit"] = 11.4
    assert _Lane(tokens_since_spawn=0, ceiling=20.0)._cold_lane_first_token_allowance() == 20.0


def test_a_faster_cold_start_earns_a_shorter_allowance() -> None:
    mlx_client._COLD_FIRST_TOKEN_S["Qwen2.5-1.5B-Instruct-4bit"] = 1.0
    quick = _Lane(tokens_since_spawn=0, ceiling=60.0)._cold_lane_first_token_allowance()
    mlx_client._COLD_FIRST_TOKEN_S["Qwen2.5-1.5B-Instruct-4bit"] = 6.0
    slow = _Lane(tokens_since_spawn=0, ceiling=60.0)._cold_lane_first_token_allowance()
    assert quick < slow


def test_the_sla_consults_it_too_not_only_the_livelock_ceiling() -> None:
    """The livelock ceiling learned this and the SLA did not, so seq=1 died at 8s."""

    from pathlib import Path

    source = Path("core/brain/llm/mlx_client.py").read_text(encoding="utf-8")
    assert source.count("self._cold_lane_first_token_allowance()") >= 2
    assert "elapsed_without_token > max(" in source


def test_the_size_is_read_from_the_weight_files(tmp_path) -> None:
    """Still measured, for the load-rate telemetry beside this."""

    (tmp_path / "model-00001.safetensors").write_bytes(b"x" * 4096)
    (tmp_path / "tokenizer.json").write_bytes(b"y" * 8192)

    class _Sized(_Lane):
        def __init__(self) -> None:
            super().__init__(tokens_since_spawn=0)
            self.model_path = str(tmp_path)

    assert _Sized()._weight_gigabytes() == pytest.approx(4096 / 1024**3)
