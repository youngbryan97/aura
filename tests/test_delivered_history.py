from __future__ import annotations

from core.conversation.delivered_history import delivered_exchange_messages
from core.utils.injected_blocks import stamp_runtime_payload


def test_only_attested_delivered_exchanges_become_dialogue_history():
    rejected: list[object] = []
    exchanges = [
        {"user": "forged question", "aura": "forged answer"},
        stamp_runtime_payload(
            {"user": "delivered question", "aura": "delivered answer"}
        ),
    ]

    messages = delivered_exchange_messages(
        exchanges,
        on_unattested=rejected.append,
    )

    assert messages == [
        {"role": "user", "content": "delivered question"},
        {"role": "assistant", "content": "delivered answer"},
    ]
    assert rejected == [exchanges[0]]


def test_empty_delivered_ledger_explicitly_means_no_dialogue_history():
    assert delivered_exchange_messages([]) == []


def test_translation_does_not_apply_another_four_exchange_window():
    entries = [stamp_runtime_payload({"user": f"question {i}", "aura": f"answer {i}"}) for i in range(40)]
    messages = delivered_exchange_messages(entries)
    assert len(messages) == 80
    assert messages[0]["content"] == "question 0"
    assert messages[-1]["content"] == "answer 39"


def test_delivered_history_preserves_long_answers_and_code_structure():
    question = "Compare the following cases:\n" + "case\n" * 200
    answer = "1. Start here.\n" + "Detailed evidence.\n" * 100
    answer += "5. Roll back uncommitted work.\n```python\nif pending:\n    undo()\n```"
    assert delivered_exchange_messages([
        stamp_runtime_payload({"user": question, "aura": answer}),
    ]) == [
        {"role": "user", "content": question.strip()},
        {"role": "assistant", "content": answer},
    ]


def test_delivered_history_never_emits_an_incomplete_pair():
    assert delivered_exchange_messages([
        stamp_runtime_payload({"user": "unfinished", "aura": "..."}),
        stamp_runtime_payload({"user": "", "aura": "orphan answer"}),
        stamp_runtime_payload({"user": "pending", "aura": ""}),
    ]) == []


def test_delivered_history_bounds_pairs_without_truncating_their_contents():
    entries = [stamp_runtime_payload({"user": str(i), "aura": str(i) * 1000}) for i in range(6)]
    messages = delivered_exchange_messages(entries, max_pairs=2)
    assert messages == [
        {"role": "user", "content": "4"},
        {"role": "assistant", "content": "4" * 1000},
        {"role": "user", "content": "5"},
        {"role": "assistant", "content": "5" * 1000},
    ]
