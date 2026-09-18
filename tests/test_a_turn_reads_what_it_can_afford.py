"""History is held to the same budget the system prompt is held to.

LIVE 2026-09-17, "Aura, what is it like to be you": 83 messages, 10,414
tokens, prefill 83.44s against decode 71.27s, for a thirty-one character
question. Forty exchanges were admitted because forty existed — the one
input to the prompt with no budget at all.
"""

from __future__ import annotations

from core.conversation.delivered_history import reached_exchange_messages
from core.conversation.history_reach import measure_reach
from core.utils.injected_blocks import stamp_runtime_payload


def _pairs(count: int, size: int = 100) -> list[dict[str, str]]:
    return [
        {"user": f"q{index:03d} " + "x" * size, "aura": f"a{index:03d} " + "y" * size}
        for index in range(count)
    ]


def _stamped(pairs):
    return [stamp_runtime_payload(dict(pair)) for pair in pairs]


def test_nothing_timed_means_nothing_cut():
    # A budget invented here would be the authored number this avoids.
    reach = measure_reach(_pairs(40), budget_chars=0)
    assert reach.retained == 40
    assert not reach.cuts_anything
    assert "nothing timed" in reach.reason


def test_a_conversation_that_fits_is_untouched():
    reach = measure_reach(_pairs(5), budget_chars=100_000)
    assert reach.boundary == 0
    assert reach.dropped == 0


def test_the_oldest_go_first_and_what_is_left_is_contiguous():
    pairs = _pairs(40)
    per_pair = len(pairs[0]["user"]) + len(pairs[0]["aura"])
    reach = measure_reach(pairs, budget_chars=per_pair * 6)
    assert reach.retained == 6, reach.reason
    assert reach.boundary == 34
    assert reach.dropped == 34
    assert reach.kept_chars <= per_pair * 6


def test_the_last_exchange_survives_a_budget_it_cannot_fit():
    # Cutting what was just said is never the right saving.
    reach = measure_reach(_pairs(40), budget_chars=1)
    assert reach.retained == 1
    assert reach.boundary == 39


def test_a_long_answer_buys_a_long_conversation():
    pairs = _pairs(40)
    per_pair = len(pairs[0]["user"]) + len(pairs[0]["aura"])
    small = measure_reach(pairs, budget_chars=per_pair * 4)
    large = measure_reach(pairs, budget_chars=per_pair * 30)
    assert large.retained > small.retained


def test_the_cut_is_said_out_loud_rather_than_hidden():
    pairs = _pairs(40)
    per_pair = len(pairs[0]["user"]) + len(pairs[0]["aura"])
    reached = reached_exchange_messages(
        _stamped(pairs), budget_chars=per_pair * 6
    )
    assert len(reached.messages) == 12
    assert "34" in reached.note
    assert "look it up" in reached.note
    assert reached.reach is not None
    assert reached.reach.to_dict()["dropped"] == 34


def test_unattested_exchanges_are_still_refused():
    dropped: list[object] = []
    reached = reached_exchange_messages(
        [{"user": "hello", "aura": "hi"}],
        budget_chars=10_000,
        on_unattested=dropped.append,
    )
    assert reached.messages == []
    assert len(dropped) == 1


def test_the_assembler_fits_the_delivered_transcript():
    from core.brain.llm.context_assembler import ContextAssembler

    pairs = _pairs(40)
    messages = []
    for pair in pairs:
        messages.append({"role": "user", "content": pair["user"]})
        messages.append({"role": "assistant", "content": pair["aura"]})
    per_pair = len(pairs[0]["user"]) + len(pairs[0]["aura"])
    kept, dropped = ContextAssembler._fit_delivered_history(
        messages, budget=per_pair * 6, estimate=len
    )
    assert len(kept) == 12
    assert dropped == 68
    # A user message leaves with the reply to it.
    assert kept[0]["role"] == "user"
    assert kept[-1]["role"] == "assistant"
    assert kept == messages[-12:]


def test_no_budget_leaves_the_assembler_transcript_whole():
    from core.brain.llm.context_assembler import ContextAssembler

    messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    kept, dropped = ContextAssembler._fit_delivered_history(
        messages, budget=0, estimate=len
    )
    assert kept == messages
    assert dropped == 0
