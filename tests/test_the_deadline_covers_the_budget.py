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
        thinking_reserve.record_decode_rate(generated_tokens=20, elapsed_s=0.5)
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


def test_the_gate_extends_for_any_user_facing_budget() -> None:
    body = _GATE.read_text()
    assert "if _is_user_facing or 0 < _answer_floor_final or _generations > 1:" in body
    assert "_decode_s = _seconds_to_decode(_tokens_to_pay_for)" in body
    assert "if _decode_s > 0.0:" in body


def test_the_extension_is_bounded_by_what_was_measured() -> None:
    body = _GATE.read_text()
    assert "(_decode_s + _read_s) * _generations" in body
    assert "timeout_val = min(_cap, _needed)" in body
    assert "if _needed > float(timeout_val):" in body


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
    assert "(_decode_s + _read_s) * _generations" in body


def test_the_generation_count_falls_back_to_one() -> None:
    body = _GATE.read_text()
    start = body.index("A turn that has to go and fetch something")
    window = body[start : start + 1200]
    assert window.count("_generations = 1") >= 2


def test_what_was_measured_survives_a_restart(tmp_path, monkeypatch) -> None:
    """The machine and the model outlive the process; the readings should too.

    LIVE, 2026-08-28: a diagnosis turn was sized for one generation because the
    rate window held no runs long enough to speak about a 640-token one. The
    runtime had been up eleven minutes and had generated dozens of times — all
    of them short, and the window is bucketed by length on purpose. Held only in
    memory, the turn that most needed the measurement was the one that never
    had it.
    """

    monkeypatch.setattr(
        thinking_reserve, "_store_path", lambda: tmp_path / "decode.json"
    )
    thinking_reserve.forget()
    for _ in range(20):
        thinking_reserve.record_decode_rate(generated_tokens=900, elapsed_s=150.0)
    thinking_reserve.record_budget_that_ran_out_thinking(budget_tokens=896)
    before = thinking_reserve.seconds_to_decode(896)
    assert before > 100.0
    assert thinking_reserve.save()

    # A new process, simulated honestly: empty memory, the file still there.
    # Calling forget() here would delete the file, which is what forget() is
    # for and the opposite of what a restart does.
    _as_if_restarted()
    # No explicit load: a restarted process takes the readings back on the
    # first question it is asked, which is the whole point of the lazy restore.
    assert thinking_reserve.seconds_to_decode(896) == pytest.approx(before, abs=1.0)
    assert thinking_reserve.reserve_tokens() == 896
    thinking_reserve.forget()


def _as_if_restarted() -> None:
    """Empty memory, disk untouched. What a new process actually sees."""

    thinking_reserve._observed.clear()
    thinking_reserve._observed_by_model.clear()
    thinking_reserve._rates.clear()
    thinking_reserve._proved_insufficient_by_model.clear()
    thinking_reserve._restored = False


def test_forgetting_removes_the_file_and_not_only_the_memory(
    tmp_path, monkeypatch
) -> None:
    """It said it forgot the disk and it did not.

    forget() cleared the queues and set _restored, which stops THIS process
    reloading and leaves the file for the next one. The test named for
    forgetting the disk was checking that one process refrained from reloading.
    """

    store = tmp_path / "decode.json"
    monkeypatch.setattr(thinking_reserve, "_store_path", lambda: store)
    thinking_reserve.forget()
    for _ in range(20):
        thinking_reserve.record_decode_rate(generated_tokens=900, elapsed_s=150.0)
    thinking_reserve.save()
    assert store.exists()

    thinking_reserve.forget()
    assert not store.exists(), "forgetting left the readings on disk"
    # And a fresh process finds nothing, which is the thing that was untrue.
    _as_if_restarted()
    assert thinking_reserve.load() == 0
    assert thinking_reserve.seconds_to_decode(896) == 0.0


def test_the_extension_reaches_the_clock_it_is_extending() -> None:
    """A number raised beside the object that holds it changes nothing.

    LIVE, 2026-08-28: "deadline 96s → 217s" was computed, logged, and the
    request expired at 98 seconds, because request_deadline was built when the
    request was admitted and never rebuilt.
    """

    body = _GATE.read_text()
    assert "timeout_val = min(_cap, _needed)" in body
    assert "request_deadline = get_deadline(float(timeout_val))" in body
    assert 'context["request_deadline_s"] = float(timeout_val)' in body
