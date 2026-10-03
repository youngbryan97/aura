"""Translate authenticated, completed exchanges into model dialogue history."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from core.conversation.history_reach import HistoryReach, measure_reach
from core.utils.injected_blocks import is_stamped_runtime_payload

VISIBLE_CONVERSATION_EXCHANGES = 40

@dataclass
class ReachedHistory:
    """The dialogue this turn reaches, and an honest note about the rest."""

    messages: list[dict[str, str]] = field(default_factory=list)
    note: str = ""
    reach: HistoryReach | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "messages": len(self.messages),
            "note": bool(self.note),
            "reach": self.reach.to_dict() if self.reach is not None else None,
        }


def _attested_pairs(
    exchanges: Any,
    *,
    max_pairs: int | None = None,
    on_unattested: Callable[[Any], None] | None = None,
) -> list[dict[str, str]]:
    """Runtime-attested user/assistant pairs that were actually delivered."""

    if not isinstance(exchanges, Sequence) or isinstance(exchanges, (str, bytes)):
        return []
    pairs: list[dict[str, str]] = []
    candidates = list(exchanges)
    if max_pairs is not None:
        candidates = candidates[-max(1, int(max_pairs)) :]
    for entry in candidates:
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
        pairs.append({"user": user_text, "aura": aura_text})
    return pairs


def _as_messages(pairs: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for pair in pairs:
        messages.append({"role": "user", "content": pair["user"]})
        messages.append({"role": "assistant", "content": pair["aura"]})
    return messages


def delivered_exchange_messages(
    exchanges: Any,
    *,
    max_pairs: int | None = None,
    on_unattested: Callable[[Any], None] | None = None,
) -> list[dict[str, str]]:
    """Return only runtime-attested user/assistant pairs already delivered."""

    return _as_messages(
        _attested_pairs(exchanges, max_pairs=max_pairs, on_unattested=on_unattested)
    )


def reached_exchange_messages(
    exchanges: Any,
    *,
    budget_chars: int = 0,
    request: str = "",
    max_pairs: int | None = None,
    on_unattested: Callable[[Any], None] | None = None,
) -> ReachedHistory:
    """The exchanges this turn can afford, rather than every exchange there is.

    Recency is not a budget. Admitting forty exchanges because forty
    exchanges exist put eighty-three seconds of prefill in front of a
    thirty-one character question, live on 2026-09-17. What is retained is
    a contiguous suffix, so nothing resolves across a hole.
    """

    pairs = _attested_pairs(
        exchanges, max_pairs=max_pairs, on_unattested=on_unattested
    )
    if not pairs:
        return ReachedHistory()

    reach = measure_reach(pairs, budget_chars=budget_chars, request=request)
    if not reach.cuts_anything:
        return ReachedHistory(messages=_as_messages(pairs), reach=reach)

    note = (
        f"[SYSTEM: this conversation has {len(pairs)} completed exchanges and "
        f"the {reach.retained} most recent are below. The other {reach.dropped} "
        "are not in front of you. If the person refers to something older, say "
        "you would need to look it up rather than reconstructing it.]"
    )
    return ReachedHistory(
        messages=_as_messages(pairs[reach.boundary :]),
        note=note,
        reach=reach,
    )


def messages_the_request_reaches(messages: Any, request: str) -> list[dict[str, str]]:
    """The contiguous suffix of a user/assistant transcript that ``request`` reaches.

    The same reach as :func:`reached_exchange_messages`, for a caller that
    holds the turn's transcript as messages rather than exchanges.
    """

    held = [dict(message) for message in messages or () if isinstance(message, dict)]
    starts = [index for index, message in enumerate(held) if message.get("role") == "user"]
    if len(starts) < 2:
        return held
    exchanges = [
        {
            "user": str(held[start].get("content") or ""),
            "aura": "\n".join(
                str(message.get("content") or "")
                for message in held[start + 1 : end]
                if message.get("role") == "assistant"
            ),
        }
        for start, end in zip(starts, [*starts[1:], len(held)], strict=True)
    ]
    reach = measure_reach(exchanges, request=request)
    return held if reach.boundary == 0 else held[starts[reach.boundary] :]


__all__ = [
    "VISIBLE_CONVERSATION_EXCHANGES",
    "ReachedHistory",
    "delivered_exchange_messages",
    "messages_the_request_reaches",
    "reached_exchange_messages",
]
