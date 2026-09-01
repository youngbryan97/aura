from __future__ import annotations

import json

from tools.run_semantic_neural_composition_canary import _reference
from tools.run_semantic_neural_composition_decode_canary import (
    ARMS,
    JOURNAL_SCHEMA,
    _append_journal_event,
    _arm_order,
    _cohort,
    _prompt_tokens,
    _state_arms,
    _wrong_state_index,
)


class _ThinkingTokenizer:
    chat_template = "composition-test-{{ enable_thinking }}"

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.encoded = ""

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append(dict(kwargs))
        channel = "<think>" if kwargs.get("enable_thinking", True) else "<answer>"
        return channel + str(messages[-1]["content"])

    def encode(self, value, *, add_special_tokens):
        assert add_special_tokens is False
        self.encoded = str(value)
        return [11, 12, 13]


def test_arm_order_is_complete_deterministic_and_rotated() -> None:
    orders = [_arm_order(f"task-{index}") for index in range(24)]

    assert all(set(order) == set(ARMS) and len(order) == len(ARMS) for order in orders)
    assert orders == [_arm_order(f"task-{index}") for index in range(24)]
    assert len({order[0] for order in orders}) == len(ARMS)


def test_structured_prefill_opens_the_answer_channel() -> None:
    tokenizer = _ThinkingTokenizer()

    assert _prompt_tokens(tokenizer, "objective", "state") == (11, 12, 13)
    assert tokenizer.calls[-1]["enable_thinking"] is False
    assert tokenizer.encoded == "<answer>objective\n\nstate"


def test_decode_journal_is_receipt_chained(tmp_path) -> None:
    journal = tmp_path / "canary.jsonl"
    journal.touch()

    first = _append_journal_event(
        journal,
        {"event": "first"},
        previous_receipt_sha256="0" * 64,
    )
    second = _append_journal_event(
        journal,
        {"event": "second"},
        previous_receipt_sha256=first,
    )
    events = [json.loads(line) for line in journal.read_text().splitlines()]

    assert [event["schema"] for event in events] == [JOURNAL_SCHEMA, JOURNAL_SCHEMA]
    assert events[1]["previous_receipt_sha256"] == first
    assert events[1]["receipt_sha256"] == second


def test_all_causal_states_share_protocol_but_not_result() -> None:
    documents = _cohort(2026083201, 2)
    all_states = [_state_arms(document) for document in documents]
    document = documents[0]
    states = all_states[0]
    expected = _reference(document)

    treatment = states["treatment"]
    wrong = all_states[_wrong_state_index(all_states, 0)]["treatment"]
    assert treatment is not None and treatment.semantic_result == expected
    assert wrong is not None and wrong.semantic_result != expected
    for arm in ("additive_lesion", "multiplicative_lesion"):
        assert states[arm] is None or states[arm].semantic_result != expected
    retained = [state for state in states.values() if state is not None]
    assert {state.report for state in retained} == {tuple(document["report"])}
    assert all(state.receipt()["teacher_available"] is False for state in retained)
