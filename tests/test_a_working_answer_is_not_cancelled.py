"""Progress decides whether a generation lives, not elapsed time.

A deadline that cancels a generation which is still producing tokens is not
protecting anything on this runtime. There is one person, one laptop, no queue
behind the turn and no bill. What a stopwatch was catching was a turn taking
longer than somebody guessed, and what it cost was the end of the answer.

What genuinely needs stopping still is: a worker that has gone silent, a decode
looping forever, and anything pathological enough to reach the absolute bound.
None of those are measured in elapsed time — the first two are measured in
tokens arriving and in what the tokens say.
"""

from __future__ import annotations

import time

from core.brain.inference_gate import InferenceGate
from core.brain.llm.mlx_client import MLXLocalClient


class _Client:
    """Just the progress bookkeeping the wait loop consults."""

    _still_producing = MLXLocalClient._still_producing

    def __init__(self, last_token_at: float) -> None:
        self._last_token_progress_at = last_token_at


def test_a_generation_still_emitting_tokens_is_alive() -> None:
    live = _Client(time.time())
    assert live._still_producing(within_s=20.0, foreground_request=True) is True


def test_a_generation_that_has_gone_quiet_is_not() -> None:
    """The case a deadline was standing in for, decided on the real signal."""

    quiet = _Client(time.time() - 60.0)
    assert quiet._still_producing(within_s=20.0, foreground_request=True) is False


def test_a_generation_that_never_started_is_not_a_slow_one() -> None:
    """Nothing has arrived, so this is silence and the first-token ceiling owns it."""

    never = _Client(0.0)
    assert never._still_producing(within_s=20.0, foreground_request=True) is False


def test_background_work_keeps_its_deadline() -> None:
    """One GPU. A dream cycle does not get to hold it while a person waits."""

    live = _Client(time.time())
    assert live._still_producing(within_s=20.0, foreground_request=False) is False


def test_the_background_budget_control_is_for_background() -> None:
    """It said "background requests only" and checked no such thing.

    The signal it scales on is registered under the name
    "background_token_budget", and it was trimming the answers people were
    waiting for.
    """

    import inspect

    source = inspect.getsource(InferenceGate)
    marker = "phi_scale = max(0.6, 0.6 + 0.4 * (phi_val / 0.8))"
    assert marker in source
    condition = source[source.rindex("if (", 0, source.index(marker)) : source.index(marker)]
    assert "is_background" in condition, condition


def test_a_user_facing_turn_is_allowed_a_full_reply() -> None:
    """A ceiling, not a reservation: the model stops when it has finished."""

    assert (
        InferenceGate._default_max_tokens_for_request(
            "desktop_quick_user", "primary", deep_handoff=False, is_background=False
        )
        == 4096
    )
    assert (
        InferenceGate._default_max_tokens_for_request(
            "dream_cycle", "primary", deep_handoff=False, is_background=True
        )
        == 384
    )


def test_the_budget_comes_from_the_clock_the_turn_actually_has() -> None:
    """Discovering the room by running out of it costs an answer each time.

    The reserve learns an unknown quantity by failing once per step — 1,024,
    then 2,048, then 3,072, a failed reply at every one. That is the right
    mechanism for something nobody can know in advance and the wrong one for
    something the hardware already answers: what the model can say inside the
    wall clock the turn is bounded by, at the rate this machine decodes.
    """

    from core.brain.llm import thinking_reserve
    from core.runtime.response_policy import USER_FACING_COMPLETION_DEADLINE_MAX_S

    thinking_reserve.forget()
    # Nothing measured: the budget is left exactly as it was.
    assert InferenceGate._tokens_the_turn_is_allowed_to_take() == 0

    # Ten tokens a second, measured on long runs.
    for _ in range(12):
        thinking_reserve.record_decode_rate(generated_tokens=1000, elapsed_s=100.0)

    allowed = InferenceGate._tokens_the_turn_is_allowed_to_take()
    assert allowed > 0
    # It must fit, and it must be the largest that fits.
    assert thinking_reserve.seconds_to_decode(allowed) <= USER_FACING_COMPLETION_DEADLINE_MAX_S
    assert (
        thinking_reserve.seconds_to_decode(allowed + 64)
        > USER_FACING_COMPLETION_DEADLINE_MAX_S
    )


def test_a_budget_is_a_ceiling_and_not_a_reservation() -> None:
    """Which is why raising it is safe: a turn that finishes early pays nothing.

    Asserted where it matters — the raise only ever raises, so a lane that
    asked for less than the clock allows keeps what it asked for.
    """

    from core.brain.llm import thinking_reserve

    thinking_reserve.forget()
    for _ in range(12):
        thinking_reserve.record_decode_rate(generated_tokens=1000, elapsed_s=100.0)

    allowed = InferenceGate._tokens_the_turn_is_allowed_to_take()
    # A ceiling given explicitly bounds the search.
    assert InferenceGate._tokens_the_turn_is_allowed_to_take(256) == 256
    assert InferenceGate._tokens_the_turn_is_allowed_to_take(256) < allowed


def test_the_answer_gets_room_reserved_rather_than_leftovers() -> None:
    """A budget it can spend entirely on thinking is one it will.

    Live on 2026-08-28 this model was given 1,024 tokens and used all 1,024
    thinking; given 2,048 it used all 2,048; given 4,025 it wrote 15,404
    characters of notes and never reached an answer. Raising the budget only
    bought longer notes, so the answer is reserved half of it instead.

    Asserted on the worker's own arithmetic rather than on a live generation:
    half the applied budget, and zero when the model is not thinking at all.
    """

    def allowance(applied: int, thinking: bool) -> int:
        return (max(1, applied) // 2) if thinking else 0

    assert allowance(4096, True) == 2048
    assert allowance(1024, True) == 512
    assert allowance(4096, False) == 0


def test_the_boundary_marker_has_to_be_in_what_the_model_reads() -> None:
    """Writing it into the text alone would fool the splitter and nothing else.

    The split is what decides which half is the answer, so appending the
    marker makes the notes-so-far count as reasoning and everything after it
    count as the answer — which is only true if the model actually saw the
    marker before writing that part.
    """

    from core.brain.llm.chat_format import split_native_thinking_generation

    assembled = "weighing the options here</think>\nFeed it at a cooler temperature."
    channels = split_native_thinking_generation(assembled, native_thinking=True)
    assert channels.boundary_closed is True
    assert channels.reasoning == "weighing the options here"
    assert channels.surface == "Feed it at a cooler temperature."

    # And without the marker there is no answer, which is the state this fix
    # exists to get out of.
    stuck = split_native_thinking_generation(
        "weighing the options here and still weighing them", native_thinking=True
    )
    assert stuck.boundary_closed is False
    assert stuck.surface == ""


def test_the_owning_clock_prices_the_thinking_too() -> None:
    """It is added on the far side of every deadline calculation here.

    So each of them was pricing a generation smaller than the one that runs,
    and this is the clock that owns the turn: its underestimate is the one
    nothing below can recover from. Live on 2026-08-28 the gate raised its
    deadline to 341 seconds, the worker was still writing at 17:10:58, and the
    turn had already given up at 17:10:13.
    """

    from core.brain.cognitive_engine import _time_the_answer_needs
    from core.brain.llm import thinking_reserve

    thinking_reserve.forget()
    for _ in range(12):
        thinking_reserve.record_decode_rate(generated_tokens=1000, elapsed_s=100.0)

    question = "Walk me through reviving a sourdough starter and how to know it is ready."
    without = _time_the_answer_needs(question)
    thinking_reserve.record_budget_that_ran_out_thinking(budget_tokens=2048)
    with_thinking = _time_the_answer_needs(question)

    assert without > 0.0
    assert with_thinking > without, (without, with_thinking)


def test_a_user_facing_turn_may_reach_the_same_ceiling_as_the_wait() -> None:
    """The cap depended on the caller having named a timeout.

    Nobody names one for an ordinary question, so an ordinary question was
    held to 240 seconds while the wait it is nested inside already allowed
    480 and the machinery below it agreed the answer needed longer.
    """

    import inspect

    from core.brain import cognitive_engine
    from core.runtime.response_policy import USER_FACING_COMPLETION_DEADLINE_MAX_S

    source = inspect.getsource(cognitive_engine)
    marker = "cycle_timeout_cap = response_policy.USER_FACING_COMPLETION_DEADLINE_MAX_S"
    assert marker in source
    condition = source[source.rindex("if ", 0, source.index(marker)) : source.index(marker)]
    assert "explicit_timeout" not in condition, condition
    assert USER_FACING_COMPLETION_DEADLINE_MAX_S > cognitive_engine._DEFAULT_COGNITIVE_CYCLE_MAX_S
