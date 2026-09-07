"""An answer budget must leave room to deliver the answer.

`_tokens_the_turn_is_allowed_to_take` searched for the largest answer that
fits the turn's wall clock and found one that fits it EXACTLY. Everything a
turn does after the last token — stabilizing, shaping, classifying,
persisting, emitting a receipt, writing the response — came out of a clock
that had already been spent.

Measured live 2026-09-07: the clock predicted 91s of reading and 148s of
decoding against a 243s deadline, the delivery path costs about 3.6s, and two
probes returned nothing at all after five minutes.

The reserve is measured in the same window and with the same pessimism as the
decode and read rates it sits beside, because a constant here is a guess about
a machine.
"""

from __future__ import annotations

from pathlib import Path

from core.brain.llm import thinking_reserve


def _reset() -> None:
    thinking_reserve._delivery_costs.clear()


def test_unmeasured_delivery_reserves_nothing() -> None:
    """Silence, like every other unmeasured quantity here."""
    _reset()
    assert thinking_reserve.seconds_to_deliver() == 0.0


def test_one_reading_is_not_a_percentile() -> None:
    _reset()
    thinking_reserve.record_delivery_cost(3.6)
    assert thinking_reserve.seconds_to_deliver() == 0.0


def test_the_reserve_is_pessimistic_once_it_is_measured() -> None:
    """The slow deliveries are the ones that lose an answer already written."""
    _reset()
    readings = [1.0] * 15 + [9.0] * 5
    for value in readings:
        thinking_reserve.record_delivery_cost(value)
    reserved = thinking_reserve.seconds_to_deliver()
    median = sorted(readings)[len(readings) // 2]
    assert reserved > median, (
        "a typical reading would miss every delivery slower than typical"
    )
    assert reserved <= max(readings), "the reserve may not exceed what was seen"


def test_a_stall_is_not_a_routine_delivery_cost() -> None:
    """One wedged turn must not size every later budget down to nothing."""
    _reset()
    for _ in range(12):
        thinking_reserve.record_delivery_cost(2.0)
    thinking_reserve.record_delivery_cost(600.0)
    assert thinking_reserve.seconds_to_deliver() <= 2.0


def test_rubbish_readings_are_ignored() -> None:
    _reset()
    for value in (float("nan"), float("inf"), -1.0):
        thinking_reserve.record_delivery_cost(value)
    assert not thinking_reserve._delivery_costs


def test_the_budget_subtracts_the_reserve() -> None:
    """The defect was a search that fit the clock exactly."""
    source = Path("core/brain/inference_gate.py").read_text()
    start = source.index("def _tokens_the_turn_is_allowed_to_take")
    end = source.index("def _reasoning_reserve", start)
    body = source[start:end]
    assert "seconds_to_deliver" in body, (
        "the budget no longer reserves anything for delivering the answer"
    )
    assert body.index("seconds_to_deliver") < body.index("low, high = 0"), (
        "the reserve must come off before the search, not after it"
    )


def test_the_route_records_what_delivery_actually_cost() -> None:
    """A reserve with no observer stays silent forever."""
    source = Path("interface/routes/chat.py").read_text()
    assert "record_delivery_cost" in source
