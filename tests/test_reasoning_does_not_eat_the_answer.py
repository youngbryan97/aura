"""A thinking model's budget is spent twice; the ceiling has to know that.

LIVE, 2026-08-27: a question about a number sequence planned at
``max_tokens=1536`` served 1,469 characters and stopped mid-paragraph, before
the part the person had asked for. Under one character per token is not prose;
the reasoning channel had taken the rest of the budget.
"""

from __future__ import annotations

import pytest

from core.brain.llm import thinking_reserve


@pytest.fixture(autouse=True)
def _clean() -> None:
    thinking_reserve.forget()
    yield
    thinking_reserve.forget()


def test_nothing_is_reserved_before_anything_is_measured() -> None:
    assert thinking_reserve.reserve_tokens() == 0
    thinking_reserve.record_reasoning_cost(
        reasoning_chars=4000, surface_chars=400, generated_tokens=1200
    )
    assert thinking_reserve.reserve_tokens() == 0


def test_the_reserve_is_what_reasoning_has_cost() -> None:
    # Four fifths of every generation went to the private channel.
    for _ in range(20):
        thinking_reserve.record_reasoning_cost(
            reasoning_chars=4000, surface_chars=1000, generated_tokens=1000
        )
    assert thinking_reserve.reserve_tokens() == 800


def test_the_ratio_comes_from_the_generation_not_from_a_constant() -> None:
    # Same character split, half the tokens: a denser tokenizer must halve the
    # reserve without an edit anywhere.
    for _ in range(20):
        thinking_reserve.record_reasoning_cost(
            reasoning_chars=4000, surface_chars=1000, generated_tokens=500
        )
    assert thinking_reserve.reserve_tokens() == 400


def test_a_generation_that_did_not_think_costs_nothing() -> None:
    for _ in range(20):
        thinking_reserve.record_reasoning_cost(
            reasoning_chars=0, surface_chars=1500, generated_tokens=400
        )
    assert thinking_reserve.reserve_tokens() == 0


def test_the_window_forgets_an_older_model() -> None:
    for _ in range(200):
        thinking_reserve.record_reasoning_cost(
            reasoning_chars=9000, surface_chars=1000, generated_tokens=1000
        )
    for _ in range(200):
        thinking_reserve.record_reasoning_cost(
            reasoning_chars=1000, surface_chars=9000, generated_tokens=1000
        )
    assert thinking_reserve.reserve_tokens() == 100


def test_rubbish_readings_are_dropped_rather_than_counted() -> None:
    for _ in range(20):
        thinking_reserve.record_reasoning_cost(
            reasoning_chars=-1, surface_chars=0, generated_tokens=0
        )
        thinking_reserve.record_reasoning_cost(
            reasoning_chars="x", surface_chars=None, generated_tokens=1
        )
    assert thinking_reserve.observations() == 0
    assert thinking_reserve.reserve_tokens() == 0


def test_the_worker_widens_the_ceiling_only_while_thinking() -> None:
    from core.brain.llm.mlx_worker import _reasoning_reserve_tokens

    assert _reasoning_reserve_tokens() == 0
    for _ in range(20):
        thinking_reserve.record_reasoning_cost(
            reasoning_chars=2000, surface_chars=2000, generated_tokens=800
        )
    assert _reasoning_reserve_tokens() == 400


def test_a_budget_that_ran_out_thinking_is_a_proof_not_a_sample() -> None:
    """One observation is enough, because waiting means every one fails first.

    The only generations that open the private channel are the ones this
    reserve exists to rescue, so a window of them can only accumulate through
    failures.
    """

    assert thinking_reserve.reserve_tokens() == 0
    thinking_reserve.record_budget_that_ran_out_thinking(budget_tokens=896)
    assert thinking_reserve.reserve_tokens() == 896


def test_the_proof_keeps_the_largest_budget_that_was_still_too_small() -> None:
    thinking_reserve.record_budget_that_ran_out_thinking(budget_tokens=512)
    thinking_reserve.record_budget_that_ran_out_thinking(budget_tokens=1280)
    thinking_reserve.record_budget_that_ran_out_thinking(budget_tokens=768)
    assert thinking_reserve.reserve_tokens() == 1280


def test_the_window_wins_when_it_measures_more_than_the_proof() -> None:
    thinking_reserve.record_budget_that_ran_out_thinking(budget_tokens=100)
    for _ in range(20):
        thinking_reserve.record_reasoning_cost(
            reasoning_chars=4000, surface_chars=1000, generated_tokens=2000
        )
    assert thinking_reserve.reserve_tokens() == 1600


def test_rubbish_proofs_are_dropped() -> None:
    thinking_reserve.record_budget_that_ran_out_thinking(budget_tokens=-5)
    thinking_reserve.record_budget_that_ran_out_thinking(budget_tokens="x")
    assert thinking_reserve.reserve_tokens() == 0


def test_forgetting_drops_the_proof_as_well_as_the_window() -> None:
    thinking_reserve.record_budget_that_ran_out_thinking(budget_tokens=2048)
    assert thinking_reserve.reserve_tokens() == 2048
    thinking_reserve.forget()
    assert thinking_reserve.reserve_tokens() == 0


def test_only_a_generation_that_thought_teaches_the_window() -> None:
    """A percentile over mostly-zeros is zero."""

    from pathlib import Path as _Path

    body = _Path("core/brain/llm/mlx_worker.py").read_text()
    start = body.index("reasoning_chars=len(native_channels.reasoning),")
    window = body[start - 700 : start]
    assert "if native_thinking is True:" in window


def test_an_answer_that_died_part_way_through_teaches_the_reserve() -> None:
    """The commoner failure, which taught it nothing for a long time.

    A generation can run out of budget while still inside the private channel
    and return no answer at all — that was recorded. It can also close the
    channel, start the answer, and die part-way through it, which is what
    happens to most questions worth thinking about. Live on 2026-08-28 a
    thinking generation spent all 1,024 of its tokens, covered neither part of
    a two-part question, and returned 1,058 characters; the reserve that
    exists to widen the budget for exactly that stood at zero.

    One proof is enough, because the generations that open the channel are the
    ones the reserve is for: waiting for a window of them means every one of
    them fails first.
    """

    from core.brain.llm import thinking_reserve

    thinking_reserve.forget()
    assert thinking_reserve.reserve_tokens() == 0

    thinking_reserve.record_budget_that_ran_out_thinking(budget_tokens=1024)
    assert thinking_reserve.reserve_tokens() == 1024
    assert thinking_reserve.proved_insufficient() == 1024

    # A later, smaller failure does not lower what has already been proved.
    thinking_reserve.record_budget_that_ran_out_thinking(budget_tokens=512)
    assert thinking_reserve.reserve_tokens() == 1024


def test_a_proof_written_by_another_process_reaches_this_one() -> None:
    """The runtime runs more than one worker, each its own process.

    The store was read once at startup and never again, so a proof paid for
    by a failed generation in one worker never reached the worker beside it
    — and never reached anything already running, which is every process that
    matters: the proof is written the moment a turn fails and wanted on the
    turn after. Live on 2026-08-28 the store held 1,024 and the generation
    that needed it was still budgeted at 1,024.
    """

    import json

    from core.brain.llm import thinking_reserve

    thinking_reserve.forget()
    assert thinking_reserve.reserve_tokens() == 0

    target = thinking_reserve._store_path()
    assert target is not None
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"proved_insufficient": 2048}))

    # No restart, no reload call: asking is enough.
    assert thinking_reserve.reserve_tokens() == 2048

    # And a store that has not changed is not read again for nothing.
    before = thinking_reserve._last_seen_store_mtime
    assert thinking_reserve.reserve_tokens() == 2048
    assert thinking_reserve._last_seen_store_mtime == before
