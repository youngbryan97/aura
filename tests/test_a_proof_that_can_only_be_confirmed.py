"""The gate that opens the private channel could not open.

`answer_is_derived_for_generation` decides whether this model call owns the
answer's computation, and so whether the private thinking channel opens. It
refused whenever the budget was no larger than `proved_insufficient` — the
largest budget any generation had ever run out of while thinking — a quantity
that only ever rose.

LIVE, 2026-09-08: it stood at 6,322 tokens for the resident 27B, set by one
runaway generation. Ordinary turns are budgeted at 512 to 1,345. Every one of
them was refused, so the model did its working in the visible answer instead:
a question about daylight at 45 degrees north was published leading with
"Roughly 105 minutes", worked its way to 210 two thousand characters below,
said "My previous 105 was a mental slip", and ran out of tokens before it
could correct the top.
"""

from __future__ import annotations

from core.brain.llm import thinking_reserve
from core.brain.llm.chat_format import answer_is_derived_for_generation

_A_MODEL = "a-27b-for-this-test"


def _forget() -> None:
    thinking_reserve.forget()


def test_one_runaway_does_not_close_the_channel_for_every_later_turn():
    _forget()
    try:
        thinking_reserve.record_budget_that_ran_out_thinking(
            budget_tokens=6322, model=_A_MODEL
        )
        assert thinking_reserve.proved_insufficient(_A_MODEL) == 6322
        # An ordinary turn: a real completion floor, an ordinary budget.
        assert answer_is_derived_for_generation(
            completion_floor=1024,
            budget_tokens=1345,
            model_name=_A_MODEL,
        ) is True
    finally:
        _forget()


def test_a_clock_with_no_time_for_the_generation_still_refuses():
    """Not gating into a coma is not the same as never gating.

    The refusal that remains is about ONE generation. The channel is a share
    of the same budget the answer is written from and the decoder closes it
    there (see core/brain/llm/a_bounded_private_channel.py), so there is no
    second cost to afford — which is what made comparing the two a deadlock:
    the clock added the reserve only once the gate said yes, and the gate said
    no because the budget had no reserve in it.
    """
    _forget()
    try:
        for _ in range(12):
            thinking_reserve.record_decode_rate(
                generated_tokens=100, elapsed_s=10.0, model=_A_MODEL
            )
        needed = thinking_reserve.seconds_to_decode(1024, _A_MODEL)
        assert needed > 0.0, "the rate window should be able to price it"
        assert answer_is_derived_for_generation(
            completion_floor=512,
            budget_tokens=1024,
            model_name=_A_MODEL,
            seconds_remaining=needed / 2.0,
        ) is False
        assert answer_is_derived_for_generation(
            completion_floor=512,
            budget_tokens=1024,
            model_name=_A_MODEL,
            seconds_remaining=needed * 2.0,
        ) is True
    finally:
        _forget()


def test_the_budget_is_not_asked_to_contain_a_reserve_nothing_has_added_yet():
    """The deadlock itself: the clock added the reserve only when this
    returned True, and this returned False because the budget had no reserve
    in it. Neither the standing proof nor the measured window decides it now."""
    import inspect

    from core.brain.llm import chat_format

    source = inspect.getsource(chat_format.answer_is_derived_for_generation)
    assert "budget <= floor + costs" not in source
    assert "budget + costs" not in source
    # The code, not the note explaining what it used to do.
    code = "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith("#")
    )
    assert "proved_insufficient" not in code
    assert "measured_reserve_tokens" not in code
    assert "THE_CHANNEL_FITS_IN" in code


def test_a_closed_question_never_opens_the_channel():
    _forget()
    try:
        assert answer_is_derived_for_generation(
            completion_floor=16,
            budget_tokens=4096,
            model_name=_A_MODEL,
        ) is False
    finally:
        _forget()


# ── and the proof itself can now come down ───────────────────────────────


def test_a_generation_that_finished_inside_a_budget_retires_the_proof():
    _forget()
    try:
        thinking_reserve.record_budget_that_ran_out_thinking(
            budget_tokens=6322, model=_A_MODEL
        )
        thinking_reserve.record_budget_that_finished_thinking(
            budget_tokens=1345, model=_A_MODEL
        )
        assert thinking_reserve.proved_insufficient(_A_MODEL) == 1344
    finally:
        _forget()


def test_the_retirement_survives_being_merged_between_processes():
    """The proof is merged as a maximum, so lowering it directly was undone by
    the next process that read the store. The two facts are kept apart and
    each is monotone in its own direction."""
    _forget()
    try:
        thinking_reserve.record_budget_that_ran_out_thinking(
            budget_tokens=6322, model=_A_MODEL
        )
        thinking_reserve.record_budget_that_finished_thinking(
            budget_tokens=1345, model=_A_MODEL
        )
        thinking_reserve._merge_reasoning_measurements(
            {"proved_insufficient_by_model": {_A_MODEL: 6322}}
        )
        assert thinking_reserve.proved_insufficient(_A_MODEL) == 1344
    finally:
        _forget()


def test_a_success_at_a_larger_budget_says_nothing_about_a_smaller_proof():
    _forget()
    try:
        thinking_reserve.record_budget_that_ran_out_thinking(
            budget_tokens=800, model=_A_MODEL
        )
        thinking_reserve.record_budget_that_finished_thinking(
            budget_tokens=4096, model=_A_MODEL
        )
        assert thinking_reserve.proved_insufficient(_A_MODEL) == 800
    finally:
        _forget()


def test_the_worker_records_both_halves_at_the_one_place_it_knows_them():
    import inspect

    from core.brain.llm import mlx_worker

    source = inspect.getsource(mlx_worker)
    at = source.index("_record_budget_that_ran_out_thinking(max_tokens, model_path)")
    nearby = source[at : at + 1400]
    assert "_record_budget_that_finished_thinking(" in nearby
    assert "native_channels.boundary_closed" in nearby
    assert "token_count < max_tokens" in nearby
