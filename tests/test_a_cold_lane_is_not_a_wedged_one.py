"""A worker that has never spoken is loading, and recycling it does not help.

LIVE 2026-08-29: a 1.5B was spawned at 02:27:07, reported ready at :12, and
was recycled at :32 for producing no token in 20.2s against a 20.0s ceiling.
The replacement was cold for the same reason. Her planner got an empty answer,
fell back to canned text, and the feed recorded that as her failure.
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

    def __init__(self, *, gigabytes: float, tokens_since_spawn: int) -> None:
        self._tokens_since_spawn = tokens_since_spawn
        mlx_client._WEIGHT_SIZES[self.model_path] = gigabytes


@pytest.fixture(autouse=True)
def _clean_host_measurements():
    sizes = dict(mlx_client._WEIGHT_SIZES)
    rates = dict(mlx_client._HOST_RATES)
    yield
    mlx_client._WEIGHT_SIZES.clear()
    mlx_client._WEIGHT_SIZES.update(sizes)
    mlx_client._HOST_RATES.clear()
    mlx_client._HOST_RATES.update(rates)


def test_a_lane_that_has_never_spoken_gets_time_to_load() -> None:
    mlx_client._HOST_RATES["weight_load"] = 0.0
    allowance = _Lane(gigabytes=1.0, tokens_since_spawn=0)._cold_lane_first_token_allowance()
    assert allowance > 0.0
    # Unmeasured, the allowance is the pessimistic read time with headroom.
    assert allowance == pytest.approx(
        (1.0 / mlx_client._UNMEASURED_WEIGHT_LOAD_GB_S) * mlx_client._COLD_START_HEADROOM
    )


def test_a_warm_lane_gets_none_of_it() -> None:
    """The allowance covers the state, not the lane."""

    assert _Lane(gigabytes=1.0, tokens_since_spawn=1)._cold_lane_first_token_allowance() == 0.0


def test_a_bigger_model_is_given_longer_to_load() -> None:
    small = _Lane(gigabytes=1.0, tokens_since_spawn=0)
    allowance_small = small._cold_lane_first_token_allowance()
    mlx_client._WEIGHT_SIZES[_Lane.model_path] = 20.0
    allowance_large = _Lane(gigabytes=20.0, tokens_since_spawn=0)._cold_lane_first_token_allowance()
    assert allowance_large > allowance_small * 10


def test_the_allowance_follows_what_the_host_was_measured_doing() -> None:
    mlx_client._HOST_RATES["weight_load"] = 4.0
    fast = _Lane(gigabytes=8.0, tokens_since_spawn=0)._cold_lane_first_token_allowance()
    mlx_client._HOST_RATES["weight_load"] = 1.0
    slow = _Lane(gigabytes=8.0, tokens_since_spawn=0)._cold_lane_first_token_allowance()
    assert slow == pytest.approx(fast * 4.0)


def test_a_model_whose_weights_cannot_be_sized_claims_nothing() -> None:
    """No measurement is not an argument for a longer ceiling."""

    assert _Lane(gigabytes=0.0, tokens_since_spawn=0)._cold_lane_first_token_allowance() == 0.0


def test_the_size_comes_from_the_weight_files(tmp_path) -> None:
    (tmp_path / "model-00001.safetensors").write_bytes(b"x" * 4096)
    (tmp_path / "tokenizer.json").write_bytes(b"y" * 8192)

    class _Sized(_Lane):
        def __init__(self) -> None:
            self._tokens_since_spawn = 0
            self.model_path = str(tmp_path)

    assert _Sized()._weight_gigabytes() == pytest.approx(4096 / 1024**3)
