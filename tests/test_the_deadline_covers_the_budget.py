"""A deadline that cannot deliver the budget is a contradiction, not a policy.

LIVE, 2026-08-27: the completion floor asked for 896 tokens and the deliberate
lane allowed about 150 seconds. At the observed decode rate 896 tokens is about
150 seconds of decoding on its own, so the generation was cut mid-thought every
time and the turn served nothing. Raising the budget alone made it worse: 1,792
tokens were granted and the clock ended it at 43 seconds.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.brain.llm import thinking_reserve

_GATE = Path("core/brain/inference_gate.py")


@pytest.fixture(autouse=True)
def _clean() -> None:
    thinking_reserve.forget()
    yield
    thinking_reserve.forget()


def test_an_unmeasured_rate_extends_nothing() -> None:
    assert thinking_reserve.seconds_to_decode(896) == 0.0
    thinking_reserve.record_decode_rate(generated_tokens=900, elapsed_s=100.0)
    assert thinking_reserve.seconds_to_decode(896) == 0.0


def test_the_time_a_budget_needs_comes_from_the_measured_rate() -> None:
    for _ in range(20):
        thinking_reserve.record_decode_rate(generated_tokens=1000, elapsed_s=100.0)
    assert thinking_reserve.seconds_to_decode(1000) == pytest.approx(100.0)


def test_only_runs_of_a_comparable_length_are_used() -> None:
    """A window of short prompts reports a rate no long turn reaches.

    LIVE, 2026-08-27: pooling them let 896 tokens look affordable inside a
    148-second turn, and the generation was cut at 98 seconds.
    """

    for _ in range(40):
        thinking_reserve.record_decode_rate(generated_tokens=40, elapsed_s=1.0)
    # Forty fast short runs and nothing long: no comparable evidence at all.
    assert thinking_reserve.seconds_to_decode(896) == 0.0
    for _ in range(12):
        thinking_reserve.record_decode_rate(generated_tokens=900, elapsed_s=150.0)
    assert thinking_reserve.seconds_to_decode(896) == pytest.approx(149.0, abs=2.0)
    # A short estimate gets to use everything it has, and the slow end of
    # that pool is still the slow long runs.
    assert thinking_reserve.seconds_to_decode(60) == pytest.approx(10.0, abs=1.0)


def test_the_slow_end_is_used_rather_than_the_typical_one() -> None:
    """A deadline sized on the typical rate misses every slower generation."""

    for _ in range(18):
        thinking_reserve.record_decode_rate(generated_tokens=400, elapsed_s=10.0)
    for _ in range(2):
        thinking_reserve.record_decode_rate(generated_tokens=500, elapsed_s=100.0)
    # 5 tok/s is the slow end; the typical rate is 40.
    assert thinking_reserve.seconds_to_decode(500) > 90.0


def test_a_rubbish_reading_is_dropped() -> None:
    for _ in range(20):
        thinking_reserve.record_decode_rate(generated_tokens=0, elapsed_s=1.0)
        thinking_reserve.record_decode_rate(generated_tokens=10, elapsed_s=0.0)
        thinking_reserve.record_decode_rate(generated_tokens="x", elapsed_s=None)
    assert thinking_reserve.seconds_to_decode(896) == 0.0


def test_forgetting_drops_the_rates() -> None:
    for _ in range(20):
        thinking_reserve.record_decode_rate(generated_tokens=100, elapsed_s=10.0)
    assert thinking_reserve.seconds_to_decode(100) > 0.0
    thinking_reserve.forget()
    assert thinking_reserve.seconds_to_decode(100) == 0.0


def test_the_gate_extends_only_when_a_floor_is_in_play() -> None:
    body = _GATE.read_text()
    start = body.index("A deadline that cannot deliver the budget")
    window = body[start : start + 2600]
    assert "if 0 < _answer_floor_final:" in window
    assert "_seconds_to_decode(max_tokens)" in window
    assert "if _decode_s > 0.0:" in window


def test_the_extension_is_bounded_by_what_was_measured() -> None:
    body = _GATE.read_text()
    start = body.index("A deadline that cannot deliver the budget")
    window = body[start : start + 3200]
    assert "(_decode_s * _generations) + _DELIVERY_MARGIN_S" in window
    assert "timeout_val = _needed" in window
    assert "if _needed > float(timeout_val):" in window


def test_the_rate_crosses_the_process_boundary() -> None:
    client = Path("core/brain/llm/mlx_client.py").read_text()
    worker = Path("core/brain/llm/mlx_worker.py").read_text()
    assert '"decode_tokens_per_second"' in worker
    assert '"decode_tokens_per_second",' in client
    assert "_carry_decode_rate_across(receipt)" in client


def test_a_turn_that_must_fetch_is_given_two_generations() -> None:
    """The call is a whole generation before the answer is started.

    LIVE, 2026-08-28: a diagnosis turn was offered the right tool, spent
    forty-five seconds emitting one call, and the request deadline expired
    fifty seconds later with nothing said about what came back. The clock
    covered one generation and the turn needed two.
    """

    body = _GATE.read_text()
    start = body.index("A turn that has to go and fetch something")
    window = body[start : start + 1200]
    assert "points_at_something_real(initial_visible_user_prompt)" in window
    assert "_generations = 2" in window
    assert "(_decode_s * _generations) + _DELIVERY_MARGIN_S" in window


def test_the_generation_count_falls_back_to_one() -> None:
    body = _GATE.read_text()
    start = body.index("A turn that has to go and fetch something")
    window = body[start : start + 1200]
    assert window.count("_generations = 1") >= 2
