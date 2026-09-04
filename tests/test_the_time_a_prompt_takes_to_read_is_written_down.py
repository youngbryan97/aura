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


def test_the_parent_writes_the_read_rate_where_the_clock_reads_it():
    from core.brain.llm import mlx_client

    source = inspect.getsource(mlx_client.MLXLocalClient._mark_token_progress)
    at = source.index("self._current_first_token_at = now")
    nearby = source[at : at + 1800]
    assert "record_read_rate(" in nearby
    assert "prompt_chars=self._current_prompt_chars" in nearby


def test_it_is_not_written_from_the_request_that_loads_the_weights():
    """Everything before the first token of a worker's life is the weights
    coming off disk as well as the prompt, which is a different fact."""
    from core.brain.llm import mlx_client

    source = inspect.getsource(mlx_client.MLXLocalClient._mark_token_progress)
    at = source.index("record_read_rate(")
    before = source[:at]
    assert '_tokens_since_spawn", 0) or 0) > 0' in before


def test_a_recorded_rate_makes_reading_cost_something():
    """The clock returns nought for an unmeasured prompt, which is what let a
    deadline be built as though reading were free."""
    thinking_reserve.forget()
    assert thinking_reserve.seconds_to_read(6000) == 0.0
    for _ in range(12):
        thinking_reserve.record_read_rate(prompt_chars=6000, elapsed_s=14.0)
    assert thinking_reserve.seconds_to_read(6000) > 0.0
    thinking_reserve.forget()
