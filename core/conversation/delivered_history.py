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
        user_text = " ".join(str(entry.get("user") or "").strip().split())[:420]
        aura_text = " ".join(str(entry.get("aura") or "").strip().split())[:520]
        if user_text:
            messages.append({"role": "user", "content": user_text})
        if aura_text and aura_text != "...":
            messages.append({"role": "assistant", "content": aura_text})
    return messages


__all__ = ["delivered_exchange_messages"]
