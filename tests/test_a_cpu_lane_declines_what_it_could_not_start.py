"""A CPU lane declines at once a prompt at least as long as one it could not start.

In seed 101's report runs, the 1.5B emergency lane on the CPU was sent her whole prompt, 25,000 to 43,000 characters,
whenever the 2-bit fallback could not run. It never produced a first token within its ceiling and returned no text
after about 64 seconds, 345 times in one run. The answer is the same when it is given at once, and on a CPU the rate
that decided it does not come and go. A GPU lane never declines, because there a stall is pressure that can clear.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from core.brain.llm import mlx_client_worker_lifecycle
from core.brain.llm.mlx_client import MLXLocalClient
from core.brain.llm.prefill_ceiling import cpu_lane_cannot_start, note_cpu_lane_could_not_start

pytestmark = pytest.mark.unit


def _lane(device: str, *, could_not_start: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        device=device,
        model_path="/models/Qwen2.5-1.5B-Instruct-4bit",
        _unstartable_from_chars=could_not_start,
        _deliberate_no_text_reason=None,
        _tokens_this_request=0,
        _current_prompt_chars=0,
    )


def _fails_to_start(lane: SimpleNamespace, chars: int) -> None:
    lane._current_prompt_chars = chars
    lane._tokens_this_request = 0
    note_cpu_lane_could_not_start(lane)


def test_a_lane_that_has_failed_nothing_declines_nothing() -> None:
    assert cpu_lane_cannot_start(_lane("cpu"), 43_000) is False


def test_a_cpu_lane_declines_a_prompt_as_long_as_one_it_could_not_start_and_tries_a_shorter_one() -> None:
    lane = _lane("cpu")
    _fails_to_start(lane, 25_517)
    assert cpu_lane_cannot_start(lane, 25_517) is True
    assert cpu_lane_cannot_start(lane, 40_355) is True
    assert cpu_lane_cannot_start(lane, 3_000) is False


def test_it_keeps_the_shortest_prompt_it_could_not_start() -> None:
    lane = _lane("cpu")
    _fails_to_start(lane, 40_000)
    _fails_to_start(lane, 25_000)
    _fails_to_start(lane, 33_000)
    assert lane._unstartable_from_chars == 25_000


def test_a_gpu_lane_never_declines_and_never_records() -> None:
    lane = _lane("gpu")
    _fails_to_start(lane, 25_517)
    assert lane._unstartable_from_chars == 0
    assert cpu_lane_cannot_start(_lane("gpu", could_not_start=1_000), 50_000) is False


def test_a_prompt_that_produced_a_token_is_not_recorded() -> None:
    lane = _lane("cpu")
    lane._current_prompt_chars = 25_517
    lane._tokens_this_request = 3
    note_cpu_lane_could_not_start(lane)
    assert lane._unstartable_from_chars == 0


def test_declining_publishes_no_deliberate_cancel() -> None:
    # The router reads a decline exactly as it read each attempt that ran out the
    # ceiling: no text from a healthy worker. A deliberate reason would change
    # the path it takes.
    lane = _lane("cpu", could_not_start=25_517)
    assert cpu_lane_cannot_start(lane, 30_000) is True
    assert lane._deliberate_no_text_reason is None


def test_the_decline_comes_before_any_request_is_built() -> None:
    source = inspect.getsource(MLXLocalClient._generate_inner)
    assert source.index("cpu_lane_cannot_start(self, len(prompt))") < source.index("uuid.uuid4()")


def test_a_new_worker_starts_with_no_record() -> None:
    source = inspect.getsource(mlx_client_worker_lifecycle)
    ready = source.index('"✅ [MLX] Worker ready: %s"')
    assert "self._unstartable_from_chars = 0" in source[ready - 300:ready]
