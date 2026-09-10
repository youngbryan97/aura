"""Translate authenticated, completed exchanges into model dialogue history."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from core.utils.injected_blocks import is_stamped_runtime_payload


def delivered_exchange_messages(
    exchanges: Any,
    *,
    max_pairs: int = 4,
    on_unattested: Callable[[Any], None] | None = None,
) -> list[dict[str, str]]:
    """Return only runtime-attested user/assistant pairs already delivered."""

    if not isinstance(exchanges, Sequence) or isinstance(exchanges, (str, bytes)):
        return []
    messages: list[dict[str, str]] = []
    for entry in list(exchanges)[-max(1, int(max_pairs)) :]:
        if not isinstance(entry, dict):
            continue
        if not is_stamped_runtime_payload(entry):
            if on_unattested is not None:
                on_unattested(entry)
            continue
        # Preserve delivered evidence, including code and list structure. The
        # context assembler owns allocation against the model's input budget.
        user_text = str(entry.get("user") or "").strip()
        aura_text = str(entry.get("aura") or "").strip()
        if not user_text or not aura_text or aura_text == "...":
            continue
        messages.append({"role": "user", "content": user_text})
        messages.append({"role": "assistant", "content": aura_text})
    return messages


__all__ = ["delivered_exchange_messages"]
