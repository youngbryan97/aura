"""A decode rate measured on one lane must not price another lane's clock.

``thinking_reserve`` keeps its readings per model and its docstring names
what happened when it did not: a 9B decodes at about fifteen tokens a second
and a 27B at half that, one window held both, and a turn on the larger model
was given a clock sized by the smaller one, ran past it, and was aborted with
everything it had written discarded — LIVE 2026-08-30, three times on one
question.

The split was made in the reserve. The two places that feed and read it
across the process boundary were not changed with it:

- ``_carry_decode_rate_across`` carries the worker's measured rate into the
  client process, and recorded it under no model at all. The resident cortex,
  the brainstem and the reflex all wrote to the one window kept for a caller
  who names nothing.
- ``_tokens_the_clock_can_deliver`` cuts a background budget the clock cannot
  pay for, and asked for the estimate with no model — so the background lane's
  budget was priced on every lane's readings at once.

That second one is what a 27B in the brainstem lane runs into. Bonsai decodes
at 8.7 tokens a second where the 9B it replaced does 21.4, and a budget cut on
the blend is a budget the lane cannot deliver.
"""
from __future__ import annotations

import pytest

from core.brain.llm import thinking_reserve
from core.brain.inference_gate import InferenceGate

FAST = "a-small-fast-checkpoint"
SLOW = "a-large-slow-checkpoint"


@pytest.fixture
def a_reserve_holding_two_lanes(monkeypatch):
    """Two models measured, one twice the speed of the other."""

    monkeypatch.setattr(thinking_reserve, "_rates", {}, raising=False)
    monkeypatch.setattr(thinking_reserve, "_restored", True, raising=False)
    monkeypatch.setattr(thinking_reserve, "_MONOTONE_CACHE", {}, raising=False)
    monkeypatch.setattr(thinking_reserve, "_written_down", lambda: None)
    for _ in range(40):
        thinking_reserve.record_decode_rate(
            generated_tokens=400, elapsed_s=20.0, model=FAST
        )   # 20 tok/s
        thinking_reserve.record_decode_rate(
            generated_tokens=400, elapsed_s=80.0, model=SLOW
        )   # 5 tok/s
    return thinking_reserve


def test_the_slow_lane_is_priced_slower_than_the_fast_one(a_reserve_holding_two_lanes):
    """The premise. Without this the rest measures nothing."""

    fast = thinking_reserve.seconds_to_decode(400, FAST)
    slow = thinking_reserve.seconds_to_decode(400, SLOW)

    assert fast > 0.0 and slow > 0.0, "both lanes were measured"
    assert slow > fast, (fast, slow)


def test_a_background_budget_is_cut_on_its_own_lanes_rate(a_reserve_holding_two_lanes):
    """The cut has to differ by lane, or naming the lane bought nothing."""

    seconds = 20.0
    on_the_fast_lane = InferenceGate._tokens_the_clock_can_deliver(
        400, seconds=seconds, model=FAST
    )
    on_the_slow_lane = InferenceGate._tokens_the_clock_can_deliver(
        400, seconds=seconds, model=SLOW
    )

    assert on_the_slow_lane < on_the_fast_lane, (on_the_fast_lane, on_the_slow_lane)
    # And the slow lane is cut below what it asked for, which is the point:
    # promising more tokens than the clock can pay for breaks mid-sentence.
    assert on_the_slow_lane < 400


def test_the_carried_rate_is_recorded_under_the_model_that_produced_it(monkeypatch):
    """The worker measures; the client records. Which model it was is the
    whole reason the reserve keeps separate windows."""

    from core.brain.llm import mlx_client

    seen: list[dict] = []
    monkeypatch.setattr(
        thinking_reserve,
        "record_decode_rate",
        lambda **kwargs: seen.append(kwargs),
    )

    mlx_client._carry_decode_rate_across(
        {"worker_verified": {"decode_tokens_per_second": 8.7}}, SLOW
    )

    assert seen, "the rate was not recorded at all"
    assert seen[0]["model"] == SLOW, seen[0]


def test_the_gate_names_the_lane_it_is_about_to_run_on():
    """`_model_now_serving` is what turns a tier into a checkpoint name, and
    a background turn runs on the brainstem rather than the cortex."""

    cortex = InferenceGate._model_now_serving("primary")
    background = InferenceGate._model_now_serving("tertiary")

    assert cortex, "the resident cortex has a name"
    assert background, "so does the background lane"
    assert cortex != background, (
        "a background turn priced on the cortex's readings is the defect this "
        "whole module is about"
    )
