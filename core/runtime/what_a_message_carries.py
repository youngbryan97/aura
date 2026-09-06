"""One envelope, so a handler is not left inferring its own context.

AutoGen's runtime was called more mature and much more locally legible than
Aura's, and one reason given was concrete: its messages have explicit
publish/send/response envelope types carrying cancellation tokens, ids and
trace metadata, and its serializer layer has a protocol with implementations
per message shape and a registry keyed by type and content type.

Aura's bus carries ``(topic: str, data: Any)``. Cluster B added what a topic
promises; this adds what a message carries. Between them a handler receives
its context instead of reconstructing it from the payload — and a handler that
infers its context has a different context depending on who called it.

One thing here is deliberately better than what it was copied from. AutoGen's
runtime notes that it saves agent state but not subscription state, so a
restarted runtime does not know who was listening. ``what_was_subscribed`` is
serialisable for exactly that reason: a subscription that does not survive a
restart is a listener that silently stops.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from core.runtime.lockdep import checked_lock
from core.runtime.what_stops_it import AnExecutionContext, Stopping

logger = logging.getLogger("Aura.WhatAMessageCarries")

__all__ = [
    "AMessage",
    "ASubscription",
    "HowItIsSent",
    "Serialiser",
    "a_reply_to",
    "for_a_type",
    "how_to_read_it",
    "how_to_write_it",
    "subscribe",
    "the_subscriptions",
    "unsubscribe",
    "what_was_subscribed",
    "who_is_subscribed_again",
]


class HowItIsSent(StrEnum):
    """The three envelopes. Nothing else is one."""

    #: To whoever is listening on the topic. No reply expected.
    PUBLISHED = "published"
    #: To one named recipient, which may answer.
    SENT = "sent"
    #: The answer to a `sent` message, carrying the id it answers.
    RESPONSE = "response"


@dataclass(frozen=True)
class AMessage:
    """One message, with everything a handler would otherwise have to infer."""

    topic: str
    payload: Any
    how: HowItIsSent = HowItIsSent.PUBLISHED
    message_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    #: The turn or tick this belongs to, so records join up across organs.
    trace: str = ""
    #: Who sent it. An authority name, not a user id.
    sender: str = ""
    #: Who it is for. Empty for a published message.
    recipient: str = ""
    #: The message this answers. Empty unless `how` is RESPONSE.
    answers: str = ""
    at: float = field(default_factory=time.time)
    #: The caller's own deadline and stop signal, carried rather than looked
    #: up. A handler that reads an ambient token gets a different token
    #: depending on which task happened to call it.
    context: AnExecutionContext | None = None

    def __post_init__(self) -> None:
        if self.how is HowItIsSent.RESPONSE and not self.answers:
            raise ValueError("a response must say which message it answers")
        if self.how is HowItIsSent.SENT and not self.recipient:
            raise ValueError("a sent message must name a recipient")
        if self.how is HowItIsSent.PUBLISHED and self.recipient:
            raise ValueError(
                "a published message goes to whoever is listening; "
                f"it cannot also name {self.recipient!r}"
            )

    @property
    def stopping(self) -> Stopping:
        """The stop signal for this message. Never None, so callers need no guard."""
        return self.context.stopping if self.context else Stopping(self.topic)

    def to_dict(self) -> dict[str, Any]:
        """Everything but the context, which is a live object and not data."""
        return {
            "topic": self.topic,
            "payload": self.payload,
            "how": str(self.how),
            "message_id": self.message_id,
            "trace": self.trace,
            "sender": self.sender,
            "recipient": self.recipient,
            "answers": self.answers,
            "at": self.at,
        }


def a_reply_to(message: AMessage, payload: Any, *, sender: str = "") -> AMessage:
    """The response to a sent message, with the ids already joined up.

    Built here rather than by each handler: the answer's ``answers`` field and
    its recipient are the two things every handler would otherwise fill in by
    hand, and by hand is where they stop matching.
    """
    if message.how is not HowItIsSent.SENT:
        raise ValueError(f"a {message.how} message expects no reply")
    return AMessage(
        topic=message.topic,
        payload=payload,
        how=HowItIsSent.RESPONSE,
        trace=message.trace,
        sender=sender or message.recipient,
        recipient=message.sender,
        answers=message.message_id,
        context=message.context,
    )


# ------------------------------------------------------------ serialisers


@runtime_checkable
class Serialiser(Protocol):
    """How one shape of payload becomes bytes and comes back."""

    content_type: str

    def write(self, payload: Any) -> bytes:
        ...

    def read(self, raw: bytes) -> Any:
        ...


class _AsJson:
    content_type = "application/json"

    def write(self, payload: Any) -> bytes:
        return json.dumps(payload, default=str, separators=(",", ":")).encode()

    def read(self, raw: bytes) -> Any:
        return json.loads(raw.decode("utf-8"))


class _AsADataclass:
    """A frozen dataclass, by its field names. Keyed by the type, not guessed."""

    content_type = "application/json+dataclass"

    def __init__(self, kind: type) -> None:
        self._kind = kind

    def write(self, payload: Any) -> bytes:
        import dataclasses

        return json.dumps(
            dataclasses.asdict(payload), default=str, separators=(",", ":")
        ).encode()

    def read(self, raw: bytes) -> Any:
        return self._kind(**json.loads(raw.decode("utf-8")))


_BY_TYPE: dict[type, Serialiser] = {}
_LOCK = checked_lock("core.runtime.what_a_message_carries", reentrant=True)


def for_a_type(kind: type, serialiser: Serialiser | None = None) -> Serialiser:
    """Register how this payload type is written and read.

    Keyed by the type. A registry keyed by a name would have to guess when two
    packages define the same name, and guessing is how a payload comes back as
    the wrong shape without anything raising.
    """
    import dataclasses

    with _LOCK:
        held = _BY_TYPE.get(kind)
        if serialiser is None:
            # Asking without saying how means "the one for this type", not
            # "replace it with a default". Building a fresh one here would let
            # an ordinary lookup silently discard a registration somebody made
            # on purpose, and a payload would come back as the wrong shape
            # with nothing raising.
            if held is not None:
                return held
            serialiser = (
                _AsADataclass(kind) if dataclasses.is_dataclass(kind) else _AsJson()
            )
        _BY_TYPE[kind] = serialiser
        return serialiser


def _serialiser_for(kind: type) -> Serialiser:
    with _LOCK:
        found = _BY_TYPE.get(kind)
    if found is not None:
        return found
    return for_a_type(kind)


def how_to_write_it(payload: Any) -> tuple[bytes, str]:
    """The bytes and the content type, from the payload's own type."""
    serialiser = _serialiser_for(type(payload))
    return serialiser.write(payload), serialiser.content_type


def how_to_read_it(raw: bytes, kind: type) -> Any:
    """Back to the shape it was, by the type asked for."""
    return _serialiser_for(kind).read(raw)


# ---------------------------------------------------------- subscriptions


@dataclass(frozen=True)
class ASubscription:
    """Who is listening to what, in a form that survives a restart."""

    topic: str
    who: str
    #: Where the handler lives, as a dotted path. A callable cannot be saved;
    #: where to find it again can.
    handler: str = ""

    def to_dict(self) -> dict[str, str]:
        return {"topic": self.topic, "who": self.who, "handler": self.handler}


_SUBSCRIPTIONS: dict[tuple[str, str], ASubscription] = {}


def subscribe(topic: str, who: str, *, handler: Any = None) -> ASubscription:
    """Record that something is listening, and where to find it again."""
    where = ""
    if handler is not None:
        module = getattr(handler, "__module__", "")
        name = getattr(handler, "__qualname__", "")
        where = f"{module}:{name}" if module and name else ""
    subscription = ASubscription(topic=str(topic), who=str(who), handler=where)
    with _LOCK:
        _SUBSCRIPTIONS[(subscription.topic, subscription.who)] = subscription
    return subscription


def unsubscribe(topic: str, who: str) -> bool:
    """Stop listening. Returns whether there was anything to stop.

    Named as its own operation rather than left to a caller clearing a dict:
    a subscription that can be added and not removed accumulates listeners
    that nothing can account for, and after a restart it comes back too.
    """
    with _LOCK:
        return _SUBSCRIPTIONS.pop((str(topic), str(who)), None) is not None


def the_subscriptions() -> tuple[ASubscription, ...]:
    with _LOCK:
        return tuple(sorted(_SUBSCRIPTIONS.values(), key=lambda one: (one.topic, one.who)))


def what_was_subscribed() -> list[dict[str, str]]:
    """The subscriptions as data, so a restart can put them back.

    AutoGen saves agent state and says it does not save this. A subscription
    that does not survive a restart is a listener that silently stops, and
    silently is the part that costs.
    """
    return [one.to_dict() for one in the_subscriptions()]


def who_is_subscribed_again(rows: Any) -> int:
    """Put subscriptions back from what was saved. Returns how many came back."""
    put_back = 0
    for row in rows or ():
        if not isinstance(row, dict) or not row.get("topic") or not row.get("who"):
            continue
        subscription = ASubscription(
            topic=str(row["topic"]),
            who=str(row["who"]),
            handler=str(row.get("handler", "")),
        )
        with _LOCK:
            _SUBSCRIPTIONS[(subscription.topic, subscription.who)] = subscription
        put_back += 1
    return put_back


def forget_everything() -> None:
    """For tests. The live runtime never calls this."""
    with _LOCK:
        _SUBSCRIPTIONS.clear()
        _BY_TYPE.clear()
