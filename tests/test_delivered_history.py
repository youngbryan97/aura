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
