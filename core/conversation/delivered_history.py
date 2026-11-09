"""Translate authenticated, completed exchanges into model dialogue history."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from core.conversation.history_reach import HistoryReach, measure_reach
from core.conversation.sitting import when
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
) -> list[dict[str, Any]]:
    """Runtime-attested user/assistant pairs that were actually delivered."""

    if not isinstance(exchanges, Sequence) or isinstance(exchanges, (str, bytes)):
        return []
    pairs: list[dict[str, Any]] = []
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
        pair = {"user": user_text, "aura": _with_what_followed(aura_text, entry.get("said_after")),
                "reply": aura_text, "remarks": _remarks(entry.get("said_after"))}
        # When it was last active, for where a sitting begins (core/conversation/sitting.py): when it was completed, or
        # when she last said something after it.
        known = [at for at in (when(entry.get("timestamp")), when(entry.get("last_said_at"))) if at]
        if known:
            pair["at"] = max(known)
        pairs.append(pair)
    return pairs


def _remarks(said_after: Any) -> list[str]:
    if not isinstance(said_after, Sequence) or isinstance(said_after, (str, bytes)):
        return []
    return [" ".join(str(words).split()) for words in said_after if str(words or "").strip()]


#: Where remarks of hers are left out of an exchange to fit the sitting, this stands in their place.
LEFT_OUT = "[…]"


def _the_sitting_fitted(pairs: list[dict[str, Any]], now: float | None = None) -> list[dict[str, Any]]:
    """``pairs`` with the sitting going on now fitted to the room it is read whole in, by leaving out remarks of hers.

    LIVE 8 November 2026, the demo: asked "So what did you think of all three?", her conversation held the sitting's
    three watchings, each an exchange of the request, her answer and what she said while it played, 24,000 characters
    in all. The sitting was over the room it is read whole in (A_SITTING_READ_WHOLE_CHARS), the request bore by meaning
    on none of it, and only the last exchange reached her: she said she had predicted "a moral panic" of a video she
    had predicted the opposite of, and thanked the video's makers for "admitting his prediction was wrong".

    What the person asked and her answers stay whole, and so does the last thing she said after each, her thoughts
    once it was over. Her remarks while it played go first, the earliest of the longest exchange each time, each run
    of them left out marked where it was.
    """
    from core.conversation.history_reach import A_SITTING_READ_WHOLE_CHARS, _exchange_chars, _this_sitting

    sitting = _this_sitting(pairs, now)
    if sitting is None:
        return pairs
    fitted = [dict(pair) for pair in pairs]
    left_out: dict[int, set[int]] = {index: set() for index in sitting}

    def written(index: int) -> str:
        pair = fitted[index]
        parts, gap = [pair["reply"]], False
        for number, remark in enumerate(pair["remarks"]):
            if number in left_out[index]:
                gap = True
                continue
            if gap:
                parts.append(LEFT_OUT)
                gap = False
            parts.append(remark)
        return "\n\n".join(parts)

    while sum(_exchange_chars(fitted[index]) for index in sitting) > A_SITTING_READ_WHOLE_CHARS:
        open_to_it = [index for index in sitting
                      if "reply" in fitted[index] and len(left_out[index]) < len(fitted[index]["remarks"]) - 1]
        if not open_to_it:
            break
        index = max(open_to_it, key=lambda at: _exchange_chars(fitted[at]))
        left_out[index].add(min(set(range(len(fitted[index]["remarks"]) - 1)) - left_out[index]))
        fitted[index]["aura"] = written(index)
    return fitted


def _with_what_followed(aura_text: str, said_after: Any) -> str:
    """Her reply and what she said after it unasked, before the person spoke again: all she said in that exchange.

    LIVE 7 November 2026, dress rehearsal: her remarks on three videos were in the chat window and not in the
    conversation her model read, whose last turn was her choosing a video before it played; asked what she thought of
    all three, she chose it again (core/conversation/said_unasked.py).
    """
    if not isinstance(said_after, Sequence) or isinstance(said_after, (str, bytes)):
        return aura_text
    followed = [" ".join(str(words).split()) for words in said_after if str(words or "").strip()]
    return "\n\n".join([aura_text, *followed]) if followed else aura_text


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
    """The exchanges this turn can afford and bears on, rather than every exchange there is.

    Recency is not a budget. Admitting forty exchanges because forty
    exchanges exist put eighty-three seconds of prefill in front of a
    thirty-one character question, live on 2026-09-17. What is retained is
    the last exchange and the earlier ones the request bears on
    (core/conversation/history_reach.py).
    """

    pairs = _attested_pairs(
        exchanges, max_pairs=max_pairs, on_unattested=on_unattested
    )
    if not pairs:
        return ReachedHistory()

    if str(request or "").strip():
        # Only a turn that answers a request. Her own turns read the sitting whole as it is, and their prompts begin
        # the same way each time, so each reuses what the last one read (core/brain/llm/prompt_cache.py).
        pairs = _the_sitting_fitted(pairs)
    reach = measure_reach(pairs, budget_chars=budget_chars, request=request)
    if not reach.cuts_anything:
        return ReachedHistory(messages=_as_messages(pairs), reach=reach)

    note = (
        f"[SYSTEM: this conversation has {len(pairs)} completed exchanges and "
        f"{reach.retained} are below: the most recent and the earlier ones this "
        f"request bears on. The other {reach.dropped} are not in front of you. If "
        "the person refers to one of them, say you would need to look it up "
        "rather than reconstructing it.]"
    )
    return ReachedHistory(
        messages=_as_messages([pairs[index] for index in reach.kept]),
        note=note,
        reach=reach,
    )


def messages_the_request_reaches(messages: Any, request: str) -> list[dict[str, str]]:
    """The exchanges of a user/assistant transcript that ``request`` reaches.

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
    if not reach.cuts_anything:
        return held
    bounds = list(zip(starts, [*starts[1:], len(held)], strict=True))
    return [message for index in reach.kept for message in held[bounds[index][0] : bounds[index][1]]]


__all__ = [
    "VISIBLE_CONVERSATION_EXCHANGES",
    "ReachedHistory",
    "delivered_exchange_messages",
    "messages_the_request_reaches",
    "reached_exchange_messages",
]
