"""One user message, one reply.

Two lanes can answer a turn. The HTTP chat route returns an answer, and the
kernel — still working on the same message through the deeper path — later
publishes its own through the event bus. Normally the route IS the consumer
of the kernel's answer and only one reaches the window. When the route has
already answered from a faster lane, the kernel's late answer arrives on its
own, minutes after the conversation moved on.

Live 2026-07-27, asked whether consciousness is just computation, she pushed
back well:

    I don't think you're right, and I'll tell you why. Consciousness isn't
    just computation — not in the way that running a program is conscious.

Three minutes later, unprompted, into the same window:

    I'll tackle this head-on. Let's break down those elements... 1.
    Decentralization - This is about distributing authority, control and
    resources across a network... blockchain or peer-to-peer networks

An answer to a question nobody asked, landing in the middle of a coherent
exchange. Earlier the same lane delivered an affect report ("More strained.
My energy level has decreased...") in reply to a question about tools.

The discrimination that matters is not "did the route answer recently" —
that would silence genuine unprompted speech, which Aura is supposed to
have. It is "did the route already answer THIS turn with something else".
So the route records what it served, and a spoken message carrying the same
answer still passes (the normal streaming path publishes the route's own
text). Only a DIFFERENT answer, arriving inside the window where it can only
be a second lane finishing the same turn, is withheld.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass
from typing import Any

from core.conversation.session_scope import (
    normalize_conversation_id,
    normalize_conversation_turn_id,
)

__all__ = [
    "LATE_LANE_WINDOW_S",
    "TURN_IN_FLIGHT_CEILING_S",
    "note_route_delivered",
    "note_turn_started",
    "route_answer_supersedes",
    "reset_route_delivery",
]

#: How long after the route answers a turn a differing spoken message is
#: treated as the other lane finishing that same turn. Chosen to cover the
#: observed gap (the deep lane trailed the route by ~3 minutes) without
#: muting proactive speech for the rest of the conversation.
LATE_LANE_WINDOW_S = 240.0

#: A turn cannot be "in flight" forever — a route that dies without answering
#: must not mute her for the rest of the session. Comfortably longer than the
#: slowest real turn (a reconstruction runs into the minutes) and far short of
#: a conversation.
TURN_IN_FLIGHT_CEILING_S = 1_800.0

@dataclass
class _RouteDeliveryState:
    conversation_id: str
    turn_id: str
    started_at: float = 0.0
    last_reply: str = ""
    last_route_at: float = 0.0


_LOCK = threading.Lock()
_STATES: dict[tuple[str, str], _RouteDeliveryState] = {}
_MAX_STATES = 256


def _norm(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def _identity(conversation_id: Any, turn_id: Any) -> tuple[str, str]:
    conversation = normalize_conversation_id(conversation_id)
    turn = normalize_conversation_turn_id(turn_id)
    if not conversation or not turn:
        raise ValueError("route delivery requires exact conversation and turn identities")
    return conversation, turn


def _prune_locked(now: float) -> None:
    stale_after = max(TURN_IN_FLIGHT_CEILING_S, LATE_LANE_WINDOW_S)
    stale = [
        key
        for key, state in _STATES.items()
        if now - max(state.started_at, state.last_route_at) > stale_after
    ]
    for key in stale:
        _STATES.pop(key, None)
    if len(_STATES) > _MAX_STATES:
        ordered = sorted(
            _STATES,
            key=lambda key: max(
                _STATES[key].started_at,
                _STATES[key].last_route_at,
            ),
        )
        for key in ordered[: len(_STATES) - _MAX_STATES]:
            _STATES.pop(key, None)


def reset_route_delivery(*, conversation_id: Any = "", turn_id: Any = "") -> None:
    """Forget all delivery state, or one exact session/turn pair."""
    with _LOCK:
        if not conversation_id and not turn_id:
            _STATES.clear()
            return
        _STATES.pop(_identity(conversation_id, turn_id), None)


def note_turn_started(*, conversation_id: Any, turn_id: Any) -> None:
    """The person just said something and is waiting for the answer.

    Protection used to begin only when the route ANSWERED, which leaves the
    whole of a turn unguarded — and the longer the turn, the wider the hole.
    Measured live 2026-07-28: asked to reverse-engineer 2048, the window filled
    with "Bryan, you mentioned her being your favorite person in the world..."
    at the same second as the real reply, and "I've been reading up on swarm
    protocols" four minutes later. Neither answered anything he asked; both are
    her own idle interests, which she is supposed to have — just not in the
    middle of someone waiting on an answer.
    """
    key = _identity(conversation_id, turn_id)
    now = time.time()
    with _LOCK:
        _prune_locked(now)
        state = _STATES.get(key)
        if state is None:
            state = _RouteDeliveryState(*key)
            _STATES[key] = state
        state.started_at = now
        state.last_reply = ""
        state.last_route_at = 0.0
        _prune_locked(now)


def note_route_delivered(
    reply_text: Any,
    *,
    conversation_id: Any,
    turn_id: Any,
) -> None:
    """Record that the chat route just answered a turn, and with what."""
    body = _norm(reply_text)
    if not body:
        return
    key = _identity(conversation_id, turn_id)
    now = time.time()
    with _LOCK:
        _prune_locked(now)
        state = _STATES.get(key)
        if state is None:
            state = _RouteDeliveryState(*key)
            _STATES[key] = state
        state.last_reply = body
        state.last_route_at = now
        _prune_locked(now)


def route_answer_supersedes(
    spoken_text: Any,
    *,
    conversation_id: Any = "",
    turn_id: Any = "",
    unprompted: bool = True,
    answering: bool = True,
) -> bool:
    """True when this spoken message is a second lane answering a settled turn.

    False when the route has not answered recently, when the window has
    passed, or when this IS the route's answer arriving through the bus —
    that last case is the normal delivery path and must never be withheld.

    ``answering`` is what the rule is actually about. An answer supersedes
    another answer; it does not supersede an event. A running commentary
    somebody asked for is a stream of things happening while the answer is
    still being worked out, and the turn stays open for as long as the work
    takes — so without this, asking her to narrate a task and then watching
    her do it produced silence for the whole of it.
    """
    body = _norm(spoken_text)
    if not body or not answering:
        return False
    now = time.time()
    conversation = normalize_conversation_id(conversation_id)
    turn = normalize_conversation_turn_id(turn_id)
    scoped = bool(conversation or turn)
    with _LOCK:
        _prune_locked(now)
        states = list(_STATES.values())
        if conversation:
            states = [state for state in states if state.conversation_id == conversation]
        if turn:
            states = [state for state in states if state.turn_id == turn]

    for state in states:
        # Only unprompted speech waits during an open turn. A direct route
        # answer travels this same bridge and must never be swallowed.
        turn_open = (
            unprompted
            and state.started_at > 0.0
            and state.started_at > state.last_route_at
            and (now - state.started_at) <= TURN_IN_FLIGHT_CEILING_S
        )
        if turn_open:
            return True
        # Every spoken EventBus publisher is required to pass OutputGate,
        # which attaches the current conversation and turn. An unscoped
        # autonomous message can still be held while any person is actively
        # waiting, but after settlement it is not evidence of a late lane and
        # must not be silenced merely because another session spoke recently.
        if not scoped:
            continue
        if not state.last_reply or (now - state.last_route_at) > LATE_LANE_WINDOW_S:
            continue
        head = body[:160]
        last_head = state.last_reply[:160]
        if head in state.last_reply or last_head in body:
            continue
        return True
    return False
