"""Close the taste loop — learn Bryan's preference from his reactions.

The conversational amplifier stashes the features of the response it sent. When Bryan's
next message arrives, we read its tone toward that response (delight vs flat vs
correction) and nudge the TasteModel accordingly. Over time the amplifier selects more
of what actually lands with him — personalized inference-time alignment, no labels asked.

Reaction signal is deliberately conservative: only clear positive/negative reactions
move the weights; neutral turns do nothing (no spurious drift).
"""
from __future__ import annotations

import logging
import re
import threading
import time
import uuid
from dataclasses import dataclass

from core.brain.taste_model import FEATURE_PRIORS, get_taste_model
from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.ConversationOutcome")

_POSITIVE = re.compile(
    r"\b(thanks?|thank you|lol|lmao|haha+|love (?:it|that|this)|exactly|perfect|nice|"
    r"great|brilliant|yes!+|that's (?:it|right|good|perfect)|well said|good (?:point|call)|"
    r"underrated|so true|real|facts)\b",
    re.I,
)
_NEGATIVE = re.compile(
    r"\b(no[.,!]?$|wrong|that's wrong|not (?:what|right|true)|nope|huh\??|that's not|"
    r"makes no sense|incoherent|generic|boring|stop (?:saying|doing)|you (?:missed|didn't)|"
    r"that doesn't|come on|ugh)\b",
    re.I,
)

#: How long a sent response stays eligible to be reacted to.
#:
#: CP126 3105e95e: register_reaction consumed the pending response for ANY
#: subsequent message — no reply relationship, no elapsed bound, no
#: intervening-event check. A "thanks" typed forty minutes later about
#: something else, or the next unrelated command, updated the taste model as
#: though it were feedback on that response. A reaction that arrives long
#: after the thing it supposedly reacts to is not a reaction.
REACTION_WINDOW_S = 600.0


@dataclass(frozen=True)
class _Pending:
    """One sent response awaiting its reply, in one conversation."""

    response_id: str
    text: str
    features: dict[str, float]
    sent_at: float
    #: What she thought of it before anyone's regard taught her anything.
    own: float = 0.0


def own_rating(features: dict[str, float]) -> float:
    """Her persona priors applied to a reply's features.

    The learned weights are what regard has already moved, so they cannot be
    what regard is measured against. See core/self/valued_for.py.
    """
    return float(
        sum(FEATURE_PRIORS.get(name, 0.0) * float(value) for name, value in (features or {}).items())
    )


_lock = threading.RLock()
#: Keyed by conversation. CP126 dea1d2f1: this was a single process-global
#: tuple with no user, session, channel or response id. Two conversations in
#: flight overwrote each other, and whichever message arrived next had its
#: tone applied to the other conversation's features — the taste model
#: learning Bryan's preferences from somebody else's reply, or from his reply
#: to a different answer.
_pending: dict[str, _Pending] = {}
_stats = {
    "recorded": 0,
    "positive": 0,
    "negative": 0,
    "neutral": 0,
    "expired": 0,
    "unmatched": 0,
}

#: Bounded so a long-lived process with many conversations cannot grow this
#: without limit; the oldest un-reacted entry is the one worth losing.
_MAX_PENDING = 64


def record_pending_response(
    text: str,
    features: dict[str, float],
    *,
    conversation_id: str = "default",
    response_id: str = "",
) -> str:
    """Remember the response just sent + its features, awaiting a reaction.

    Returns the response id, so the caller can prove which response a later
    reaction is about rather than relying on ordering.
    """
    global _pending
    with _lock:
        identifier = str(response_id or f"resp-{uuid.uuid4().hex[:12]}")
        _pending[str(conversation_id or "default")] = _Pending(
            response_id=identifier,
            text=str(text or ""),
            features=dict(features or {}),
            sent_at=time.monotonic(),
            own=own_rating(features or {}),
        )
        _stats["recorded"] += 1
        while len(_pending) > _MAX_PENDING:
            oldest = min(_pending, key=lambda key: _pending[key].sent_at)
            del _pending[oldest]
        return identifier


def _classify_reward(user_message: str) -> float:
    msg = str(user_message or "").strip()
    if not msg:
        return 0.0
    pos = bool(_POSITIVE.search(msg))
    neg = bool(_NEGATIVE.search(msg))
    if pos and not neg:
        return 1.0
    if neg and not pos:
        return -1.0
    return 0.0


def register_reaction(
    user_message: str,
    *,
    conversation_id: str = "default",
    in_reply_to: str = "",
) -> float | None:
    """Judge the reaction to the pending response and update the taste model.

    Returns the reward applied (+1/-1), or None if there was nothing to learn
    from — including when the message is not plausibly a reaction at all.

    A reaction must belong to the SAME conversation, and must arrive inside
    ``REACTION_WINDOW_S``. When the caller can name the response being replied
    to, that is checked too; when it cannot, adjacency within the window is
    the weaker evidence this loop runs on, and a mismatch is refused rather
    than guessed at.
    """
    global _pending
    key = str(conversation_id or "default")
    with _lock:
        pending = _pending.pop(key, None)
        if pending is None:
            _stats["unmatched"] += 1
            return None
        if time.monotonic() - pending.sent_at > REACTION_WINDOW_S:
            _stats["expired"] += 1
            return None
        if in_reply_to and in_reply_to != pending.response_id:
            # The caller named a DIFFERENT response. Putting it back would be
            # wrong too — that response has been superseded — but crediting
            # this message to it would be worse.
            _stats["unmatched"] += 1
            return None
    features = pending.features
    if not features:
        return None
    reward = _classify_reward(user_message)
    if reward == 0.0:
        _stats["neutral"] += 1
        return None
    # Regard that goes to what she values in a reply teaches her taste at the
    # full rate; regard that goes to something else about it teaches less, so
    # she is not remade around the token. See core/self/valued_for.py.
    try:
        from core.self.valued_for import get_valued_for_ledger

        ledger = get_valued_for_ledger()
        ledger.note(pending.own, reward)
        lesson = reward * ledger.weight()
    except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
        record_degradation("conversation_outcome_valued_for", exc, action="taught the full reaction")
        lesson = reward
    try:
        get_taste_model().update(features, lesson)
    except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
        record_degradation("conversation_outcome_update", exc)
        return None
    _stats["positive" if reward > 0 else "negative"] += 1
    logger.info("🗣️ [TasteLoop] reaction reward=%+.0f applied to taste model.", reward)
    return reward


def stats() -> dict[str, int]:
    with _lock:
        return dict(_stats)


def reset() -> None:
    global _pending
    with _lock:
        _pending = {}
