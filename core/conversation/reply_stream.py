"""core/conversation/reply_stream.py — the governed reply, while it is still forming.

A surface that has to wait for the finished string pays the whole reply's
latency before its first byte. For text that is invisible; for speech it is
the dominant term, because nothing can be *said* until everything has been
*decided*. The voice lane's answer to that was a 45-word cap — latency
management wearing the costume of a style choice, and the reason spoken
answers were shallower than the same question typed.

The governed pipeline already produces its reply incrementally: the state
machine emits ``chat_stream_chunk`` as tokens land. What was missing is a way
for one surface to receive *its own* turn's chunks. The telemetry topic those
chunks ride on is global, so a subscriber cannot tell its turn from another
surface's — and a voice lane that spoke the desktop's reply would be a far
worse defect than a slow one.

So this is a channel bound to the *async context* of a turn, not to a topic.
``bind_reply_stream`` sets a ContextVar; the publish site walks no registry
and takes no lock, it simply asks what channel this turn is running under.
Context propagates into every await and is copied into every task spawned
underneath, so the binding follows the turn through the whole call chain and
cannot leak sideways into a concurrent one. A turn with nothing bound — every
text turn, today — takes an ``is None`` check and moves on.

**Publishing never blocks cognition.** The queue is bounded and writes are
non-blocking. A consumer that stalls (a synthesiser under back-pressure, a
socket that stopped draining) loses chunks and is told it lost them; it does
not slow down the mind producing them. That is the right trade in both
directions: the stream is an accelerator, and the finished string is still
delivered by the ordinary path.

**The stream is not the authority.** Chunks are pre-stabilisation text. The
governed reply that the turn finally returns is what Aura is accountable for.
``reconcile`` exists to say what a consumer must do when the two disagree,
and the answer is never "quietly continue".
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.Conversation.ReplyStream")

# Bounded so a stalled consumer costs chunks rather than memory. A governed
# reply arrives as clause-sized pieces; 256 of them is far more than any real
# turn produces, so hitting this bound means the consumer has stopped, not
# that the reply was long.
_MAX_PENDING_CHUNKS = 256

# How long to wait for the next chunk before concluding the producer has
# stopped rather than merely thinking. Deliberately generous: a hard question
# can leave a real gap between clauses, and cutting a reply short because the
# model paused would be a worse failure than waiting. What this rules out is
# the unbounded case — a producer that dies without closing its channel
# leaving the consumer parked forever, holding a speaking track and a session
# state with it.
DEFAULT_CHUNK_TIMEOUT_S = 120.0

_RECOVERABLE = (RuntimeError, AttributeError, TypeError, ValueError)


@dataclass(slots=True)
class StreamStats:
    """What the channel actually carried, for honest reporting afterwards."""

    published: int = 0
    dropped: int = 0
    consumed: int = 0

    @property
    def lossless(self) -> bool:
        return self.dropped == 0


class ReplyStreamChannel:
    """One turn's reply, delivered as it forms.

    Iterating the channel yields chunks until the producing turn closes it.
    Closing is mandatory and idempotent: a consumer blocked on a channel whose
    producer died would be a hang, so every path that binds one closes it.
    """

    __slots__ = ("_queue", "_closed", "_stats", "_loop")

    def __init__(self) -> None:
        self._queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=_MAX_PENDING_CHUNKS)
        self._closed = False
        self._stats = StreamStats()
        # Captured at bind time. The publish site may be called from a worker
        # thread (the state machine's telemetry hop is sync), and a Queue may
        # only be touched from its own loop.
        self._loop: asyncio.AbstractEventLoop | None = None

    # ── producer side ────────────────────────────────────────────────────

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def publish(self, chunk: str) -> None:
        """Offer one chunk. Never blocks, never raises into cognition."""
        text = str(chunk or "")
        if not text or self._closed:
            return
        try:
            self._queue.put_nowait(text)
        except asyncio.QueueFull:
            self._stats.dropped += 1
            if self._stats.dropped == 1:
                # Once per turn. A consumer that has stopped draining will
                # trip this on every subsequent chunk, and a degradation
                # record per chunk would bury the one that matters.
                record_degradation(
                    "conversation.reply_stream",
                    RuntimeError("reply stream consumer stopped draining"),
                    action=(
                        "dropped streamed chunks; the finished reply is still "
                        "delivered by the ordinary path"
                    ),
                    severity="warning",
                )
            return
        self._stats.published += 1

    def publish_threadsafe(self, chunk: str) -> None:
        """Publish from a thread that is not running the channel's loop."""
        loop = self._loop
        if loop is None or self._closed:
            return
        try:
            running = asyncio.get_running_loop()
        # not a failure: off a loop there is no running loop, so the threadsafe path below
        # is the one that applies.
        except RuntimeError:
            running = None
        if running is loop:
            self.publish(chunk)
            return
        with contextlib.suppress(RuntimeError):
            loop.call_soon_threadsafe(self.publish, chunk)

    def close(self) -> None:
        """End the stream. Idempotent; a consumer is always released."""
        if self._closed:
            return
        self._closed = True
        with contextlib.suppress(asyncio.QueueFull):
            self._queue.put_nowait(None)

    def close_threadsafe(self) -> None:
        loop = self._loop
        if loop is None:
            self.close()
            return
        try:
            running = asyncio.get_running_loop()
        # not a failure: the same: no running loop here.
        except RuntimeError:
            running = None
        if running is loop:
            self.close()
            return
        with contextlib.suppress(RuntimeError):
            loop.call_soon_threadsafe(self.close)

    # ── consumer side ────────────────────────────────────────────────────

    async def __aiter__(self) -> AsyncIterator[str]:
        """Iterate with the default budget. See ``drain`` for an explicit one.

        Bounded rather than a bare ``queue.get()``: a producer that dies
        without closing would otherwise park its consumer forever, and on the
        voice path that consumer is holding a speaking track and a session
        state. The budget is generous because a slow clause is normal and only
        a *stopped* producer is a fault.
        """
        async for chunk in self.drain(timeout_s=DEFAULT_CHUNK_TIMEOUT_S):
            yield chunk

    async def drain(self, *, timeout_s: float) -> AsyncIterator[str]:
        """Iterate with a deadline on each individual chunk.

        A producer that wedges mid-reply must not wedge the surface waiting on
        it. Exhausting the timeout ends the stream; the caller still has the
        finished reply to fall back on, and ``stats.consumed`` tells it how
        much of the answer it already committed to out loud.
        """
        while True:
            # ``close`` may be unable to enqueue its sentinel while the queue
            # is full. Once the already-published chunks are drained, the
            # closed state itself is terminal evidence; do not wait for a
            # sentinel that backpressure legitimately prevented.
            if self._closed and self._queue.empty():
                return
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=max(0.05, timeout_s))
            except TimeoutError:
                record_degradation(
                    "conversation.reply_stream",
                    TimeoutError(f"no reply chunk within {timeout_s:.1f}s"),
                    action="ended the streamed reply; the finished reply still arrives",
                    severity="warning",
                )
                return
            if item is None:
                return
            self._stats.consumed += 1
            yield item

    @property
    def stats(self) -> StreamStats:
        return self._stats

    @property
    def closed(self) -> bool:
        return self._closed


# The binding is per-async-context, which is what makes this safe to publish
# into from a shared pipeline: a turn can only ever reach its own channel.
_ACTIVE_STREAM: ContextVar[ReplyStreamChannel | None] = ContextVar(
    "aura_active_reply_stream", default=None
)


@contextmanager
def bind_reply_stream(channel: ReplyStreamChannel) -> Iterator[ReplyStreamChannel]:
    """Bind a channel for the duration of one governed turn."""
    with contextlib.suppress(RuntimeError):
        channel.attach_loop(asyncio.get_running_loop())
    token = _ACTIVE_STREAM.set(channel)
    try:
        yield channel
    finally:
        _ACTIVE_STREAM.reset(token)
        channel.close()


def active_reply_stream() -> ReplyStreamChannel | None:
    return _ACTIVE_STREAM.get()


def publish_reply_chunk(chunk: str) -> None:
    """Publish to whatever channel this turn is bound to, if any.

    Called from the cognition path. The overwhelming majority of turns have
    nothing bound, so the hot path is one ContextVar read.
    """
    channel = _ACTIVE_STREAM.get()
    if channel is None:
        return
    try:
        channel.publish_threadsafe(chunk)
    except _RECOVERABLE as exc:
        record_degradation(
            "conversation.reply_stream",
            exc,
            action="dropped one streamed chunk; cognition was not disturbed",
            severity="debug",
        )


# ── reconciliation ───────────────────────────────────────────────────────


@dataclass(slots=True)
class Reconciliation:
    """How a spoken prefix relates to the reply the turn finally returned.

    The streamed text is what the listener already heard. The final text is
    what the governed turn stands behind. Three outcomes, and only one of them
    is "carry on":

    ``consistent`` — the spoken text is a prefix of the final reply. Speak the
    remainder and the turn is whole.

    ``diverged`` — governance changed something that was already said out
    loud. This is not recoverable by silence: the listener is holding a
    sentence Aura no longer stands behind, and the only honest move is to say
    so. ``correction_context`` is what her next turn is told about it.

    ``empty`` — the stream carried nothing, which is the normal case for any
    turn that did not take the streaming path. Speak the final reply as usual.
    """

    status: str
    remainder: str = ""
    spoken: str = ""
    final: str = ""
    correction_context: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def consistent(self) -> bool:
        return self.status == "consistent"

    @property
    def diverged(self) -> bool:
        return self.status == "diverged"

    @property
    def empty(self) -> bool:
        return self.status == "empty"


def _normalize(text: str) -> str:
    """Collapse whitespace so formatting differences are not divergence."""
    return " ".join(str(text or "").split())


def reconcile(spoken: str, final: str) -> Reconciliation:
    """Decide what a surface owes the listener after a streamed reply.

    Whitespace and trailing punctuation are not divergence — the stabiliser
    re-flows text and that changes nothing anyone heard. A different *word* is
    divergence, and it is reported as such however small it looks.
    """
    spoken_n = _normalize(spoken)
    final_n = _normalize(final)

    if not spoken_n:
        return Reconciliation(status="empty", remainder=final_n, spoken="", final=final_n)

    if not final_n:
        # The turn spoke and then returned nothing to stand behind. Treat it
        # as divergence rather than success: something failed after delivery.
        return Reconciliation(
            status="diverged",
            spoken=spoken_n,
            final="",
            correction_context=(
                "[voice context: you spoke a reply out loud, but the governed turn "
                "returned no final text to stand behind. The user heard: "
                f'"{spoken_n[:400]}". Say plainly that the rest of that answer did '
                "not survive your own checks, rather than continuing as if it had.]"
            ),
            notes=["final reply was empty after audio had already been delivered"],
        )

    if final_n.startswith(spoken_n):
        remainder = final_n[len(spoken_n):].strip()
        return Reconciliation(
            status="consistent", remainder=remainder, spoken=spoken_n, final=final_n
        )

    # Case-insensitive prefix match: the stabiliser sometimes re-capitalises a
    # sentence opener, which no listener would call a different answer.
    if final_n.lower().startswith(spoken_n.lower()):
        remainder = final_n[len(spoken_n):].strip()
        return Reconciliation(
            status="consistent",
            remainder=remainder,
            spoken=spoken_n,
            final=final_n,
            notes=["prefix matched only after case folding"],
        )

    return Reconciliation(
        status="diverged",
        spoken=spoken_n,
        final=final_n,
        correction_context=(
            "[voice context: your reply was revised by your own governance after "
            "part of it had already been spoken. The user heard: "
            f'"{spoken_n[:400]}". The reply you actually stand behind is: '
            f'"{final_n[:400]}". Correct the record in your own words — say what '
            "you got wrong, then give the answer you stand behind. Do not repeat "
            "the whole thing as if nothing happened.]"
        ),
        notes=["streamed prefix is not a prefix of the final governed reply"],
    )


__all__ = [
    "Reconciliation",
    "ReplyStreamChannel",
    "StreamStats",
    "active_reply_stream",
    "bind_reply_stream",
    "publish_reply_chunk",
    "reconcile",
]
