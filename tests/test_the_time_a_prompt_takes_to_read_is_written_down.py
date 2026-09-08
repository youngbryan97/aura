"""A measurement taken in one process is no use to a deadline built in another.

The worker records how fast it read a prompt. The deadlines are built in the
parent, from a record that lives per-process — so on a fresh runtime the answer
clock believed reading a prompt was free.

LIVE 2026-09-04, forty minutes after a clean boot: the clock granted 23
seconds, the same worker measured the same prompt as needing 23.2 to read, and
every user-facing generation was cancelled at 23. The fallback ladder then
found no small model admitted under the memory headroom, waited out its budget,
and the turn ended in a refusal. Five in ten minutes, no answer delivered.
"""

from __future__ import annotations

import inspect

from core.brain.llm import thinking_reserve


def test_the_read_rate_is_written_from_the_time_mlx_spent_reading():
    """The one clock that measures reading, and it is not the one that runs
    until the first token.

    LIVE, 2026-09-08: the stored window had settled at about 14 characters a
    second, because a read rate was being recorded from first-token latency —
    which contains the reading and also the queue, the cache build, the sampler
    and the weights coming off disk. The answer clock believed a 9,558-character
    prompt would take 690 seconds to read and sized the turn at 893. The worker
    read it at 410 to 990 tokens a second and the answer came back in 64.
    """
    from core.brain.llm import mlx_worker

    source = inspect.getsource(mlx_worker)
    at = source.index("_record_read_rate(_prompt_chars_for_rate, _read_s)")
    nearby = source[max(0, at - 2000) : at + 200]
    assert 'generation_performance.get("prefill_seconds")' in nearby
    assert "_record_read_rate(_prompt_chars_for_rate, _read_s)" in source


def test_no_read_rate_is_written_when_mlx_did_not_time_the_prompt():
    """An unmeasured rate extends no deadline — this module's own discipline.

    Substituting first-token latency for a missing measurement is what put the
    wrong readings in the window, and a stored reading outlives the turn that
    produced it.
    """
    from core.brain.llm import mlx_worker

    source = inspect.getsource(mlx_worker)
    at = source.index("_record_read_rate(_prompt_chars_for_rate, _read_s)")
    before = source[max(0, at - 1200) : at]
    assert "if _read_s > 0.0:" in before
    assert "first_token_latency_s" not in before.split("_read_s = 0.0")[-1]


def test_readings_taken_with_the_other_clock_are_not_read_back():
    """They are not slow readings; they are measurements of another quantity."""
    from core.brain.llm import thinking_reserve as reserve

    assert reserve._READ_RATE_KEY != "read_rates"
    source = inspect.getsource(reserve)
    assert source.count('"read_rates"') == 0


def test_a_recorded_rate_makes_reading_cost_something():
    """The clock returns nought for an unmeasured prompt, which is what let a
    deadline be built as though reading were free."""
    thinking_reserve.forget()
    assert thinking_reserve.seconds_to_read(6000) == 0.0
    for _ in range(12):
        thinking_reserve.record_read_rate(prompt_chars=6000, elapsed_s=14.0)
    assert thinking_reserve.seconds_to_read(6000) > 0.0
    thinking_reserve.forget()


# ── and the deadline uses the number the worker judges itself by ─────────


def test_the_worker_says_how_long_its_own_prompt_takes():
    from core.brain.llm import mlx_client

    assert callable(getattr(mlx_client.MLXLocalClient, "least_time_to_read", None))


def test_the_answer_clock_asks_the_worker_that_will_serve_it():
    """Two estimates of one fact are two deadlines, and the smaller one wins
    by cancelling the work.

    LIVE 2026-09-04, one line apart: "the prompt takes about 2s to read",
    granting 25 seconds, and "a 2867-char prompt takes about 8.8s to read at
    82 tok/s", needing 26.3.
    """
    from core.brain import inference_gate

    source = inspect.getsource(inference_gate)
    at = source.index("_read_s = _seconds_to_read(_prompt_chars_for_clock)")
    nearby = source[at : at + 1400]
    assert "least_time_to_read" in nearby
    assert "_read_s = max(_read_s, _worker_says)" in nearby


def test_a_worker_that_cannot_say_leaves_the_clock_as_it_was():
    from core.brain import inference_gate

    source = inspect.getsource(inference_gate)
    at = source.index("_worker_says = 0.0")
    assert "if callable(_knows):" in source[at : at + 400]
