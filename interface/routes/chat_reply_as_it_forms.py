"""A reply as it forms, shown in the window it was asked from: what she took a message for as soon as it is read, and her
words as they are written, a checked sentence at a time.

LIVE 24 November 2026, over 43 turns of a day, the median reply took 111 s and was shown when the last of it was written. Its
first words were ready some thirty seconds in. The voice has streamed them since November (core/conversation/reply_stream.py);
this gives a typed turn the same stream, to the one window that asked, by the key it asked with.

What is shown is what has passed the checks each sentence passes alone. The whole reply is checked as it was, and takes the
place of what was shown.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from core.conversation.reply_stream import (
    ReplyStreamChannel,
    active_reply_stream,
    bind_reply_stream,
)
from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.Chat")

__all__ = ["reply_as_it_forms", "tell_the_window"]

#: How often what is written is sent, in seconds: as often as is seen as one movement.
SENT_EVERY_S = 0.08
#: The window the turn running here was asked from, by the key it asked with.
_KEY: ContextVar[str] = ContextVar("aura_window_of_this_turn", default="")

#: What she is doing about a message, by what it was read as asking for, as a person would say it.
DOING = {"talk": "Listening", "known": "Thinking it over", "computed": "Working it out", "found": "Looking into it", "program": "Getting started on it",
         "program_changed": "Getting started on it", "document_read": "Reading it", "written": "Writing", "status": "Checking how it's going", "resume": "Picking it back up",
         "on_machine": "Getting to it", "sent": "Getting it ready", "kept": "Making a note of it"}


def _send(**facts: Any) -> None:
    from core.event_bus import get_event_bus

    try:
        get_event_bus().publish_threadsafe("telemetry", {"timestamp": time.time(), "type": "aura_turn", **facts})
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation("chat.reply_as_it_forms", exc, severity="debug", action="a reply was shown when it was whole, and not as it formed")


def tell_the_window(**facts: Any) -> None:
    """Tell the window this turn was asked from something of it; nothing where it was asked from no window."""
    key = _KEY.get()
    if key:
        _send(key=key, **facts)


class _ToTheWindow(ReplyStreamChannel):
    """A turn's reply, sent to its window as it forms. Nobody takes from it, so nothing waits in it."""

    __slots__ = ("_key", "_sent_at", "_sent")

    def __init__(self, key: str) -> None:
        super().__init__()
        self._key, self._sent_at, self._sent = key, 0.0, ""

    def publish(self, chunk: str) -> None:
        text = str(chunk or "")
        if not text or self._closed:
            return
        self._published.append(text)
        now = time.monotonic()
        if now - self._sent_at >= SENT_EVERY_S:
            self._sent_at, self._sent = now, self.published_text()
            _send(key=self._key, text=self._sent)

    def close(self) -> None:
        if not self._closed and self._published and self.published_text() != self._sent:
            _send(key=self._key, text=self.published_text())
        super().close()


@contextmanager
def reply_as_it_forms() -> Iterator[None]:
    """For the turn running here: shown as it forms to the window that asked for it, where one did and nothing has its words already."""
    from core.runtime.chat_delivery_progress import current_chat_delivery_identity

    identity = current_chat_delivery_identity()
    key = str((identity or {}).get("idempotency_key") or "")
    if not key or active_reply_stream() is not None:
        yield
        return
    token = _KEY.set(key)
    try:
        with bind_reply_stream(_ToTheWindow(key)):
            yield
    finally:
        _send(key=key, done=True)
        _KEY.reset(token)
