"""The append-only record behind the live conversation window.

What this closes
----------------
:mod:`core.context.condenser` made forgetting a replayable event, and then
nothing called it. The live path was, and is, ``_prune_history_async`` in
:mod:`core.orchestrator.mixins.context_streaming`: it hands the history to
``context_pruner``, which drops everything but the last thirty messages and puts
a ``[CONSOLIDATED MEMORY]`` summary in front, and the result is assigned back
over ``self.conversation_history``. The messages that left are gone from the
process at that moment. The only trace is a log line with two counts.

So you cannot ask what she could see three turns ago, you cannot replay the
window a decision was made in, and when a long conversation goes wrong there is
no way to tell whether the answer was bad or the context was.

This module keeps the log and records the forgetting. The window the model sees
is unchanged — that is the point, and :meth:`ConversationLog.record_pruning`
asserts it by deriving the window back out of the log and comparing. What is
added is the ability to reconstruct any earlier one.

Why it wraps the existing pruner rather than replacing it
---------------------------------------------------------
``context_pruner`` decides *what* to forget, and that decision is tuned, live
and load-bearing. Swapping it for one of the condenser strategies would change
what she remembers in the same change that adds the record, and if her replies
shifted afterwards there would be no way to tell which half did it.

So the strategy is untouched. This observes the transition — before and after —
and expresses it in the condenser's vocabulary. The
:class:`~core.context.condenser.Condensation` it derives is the same object the
condenser strategies produce, and :meth:`~core.context.condenser.View.from_events`
is what replays it, so the two describe forgetting the same way and a strategy
swap later is a swap of one call.

Alignment
---------
The pruner returns a list, not a diff, so the correspondence has to be
recovered. It only ever drops messages and inserts a summary at the front,
which means the kept messages are a *subsequence* of what went in. Alignment
walks both in order — the standard subsequence match — and anything in the
result that is not in the input is a newly written summary. That holds for the
fallback path too, where the pruner returns a plain tail slice.

If the alignment cannot be trusted — the result is not a subsequence, the
identity check fails — the log records a break rather than a condensation and
resynchronises. A record that quietly drifts out of step with the live history
is worse than one that says it lost the thread, because it looks like evidence.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from core.context.condenser import Condensation, ContextEvent, View, estimate_tokens
from core.runtime.lockdep import checked_lock

logger = logging.getLogger("Aura.ConversationLog")

__all__ = [
    "LoggedMessage",
    "ConversationLog",
    "get_conversation_log",
]

#: Messages retained in the append-only log.
#:
#: The log outlives the window on purpose, so this is the bound that stops it
#: outliving the process's memory. It is deliberately far above any live window:
#: the pruner keeps thirty messages, so this holds the history behind roughly
#: two hundred prunings. Eviction is from the oldest end and is itself recorded,
#: so the log never silently becomes partial.
DEFAULT_LOG_CAPACITY = 6000

#: Condensations retained. Each is small — ids, a summary, a reason — but a
#: long-lived process condenses indefinitely.
DEFAULT_CONDENSATION_CAPACITY = 512


def _role_of(message: Mapping[str, Any]) -> str:
    return str(message.get("role") or "unknown")


def _content_of(message: Mapping[str, Any]) -> str:
    content = message.get("content")
    return content if isinstance(content, str) else str(content or "")


def _same(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    """Whether two messages are the same message.

    Role and content only. The live history carries incidental keys — timestamps,
    routing hints — that the pruner is free to drop or add while passing the
    message through, and treating those as identity would make every prune look
    like a total replacement.
    """
    return _role_of(a) == _role_of(b) and _content_of(a) == _content_of(b)


@dataclass(frozen=True)
class LoggedMessage:
    """One message, with the identity the log refers to it by."""

    event_id: int
    message: dict[str, Any]

    @property
    def role(self) -> str:
        return _role_of(self.message)

    @property
    def content(self) -> str:
        return _content_of(self.message)

    def as_event(self, *, pinned: bool = False) -> ContextEvent:
        return ContextEvent(
            event_id=self.event_id,
            kind=self.role,
            content=self.content,
            pinned=pinned,
        )


@dataclass
class ConversationLog:
    """Every message the conversation has held, and every forgetting of it."""

    capacity: int = DEFAULT_LOG_CAPACITY
    condensation_capacity: int = DEFAULT_CONDENSATION_CAPACITY

    _messages: dict[int, LoggedMessage] = field(default_factory=dict)
    _order: list[int] = field(default_factory=list)
    _live: list[int] = field(default_factory=list)
    _condensations: list[Condensation] = field(default_factory=list)
    #: The highest message id in existence when each condensation was recorded.
    #:
    #: Without it, reconstruction applies old condensations to the log as it
    #: stands *now*, so the "window from three turns ago" arrives carrying every
    #: message said since. That is not the window she saw, and it is the exact
    #: kind of plausible-but-wrong record this module is supposed to replace.
    _watermarks: list[int] = field(default_factory=list)
    _next_id: int = 1
    _next_summary_id: int = -1
    _evicted: int = 0
    _breaks: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    # -- recording -------------------------------------------------------

    def observe(self, history: Sequence[Mapping[str, Any]]) -> None:
        """Take up the current live history, assigning ids to anything new.

        Called before pruning, and safe to call on every turn. Messages already
        known keep their ids; the tail that has arrived since last time gets new
        ones. This is what lets the log be attached to a conversation already in
        progress without pretending to know what came before.
        """
        with self._lock:
            self._observe_locked(history)

    def _observe_locked(self, history: Sequence[Mapping[str, Any]]) -> None:
        aligned, unmatched = self._align_locked(self._live, history)
        if unmatched:
            # New messages since the last observation. They are appended in the
            # order they appear, which is the order they were said.
            pass
        self._live = aligned

    def record_pruning(
        self,
        before: Sequence[Mapping[str, Any]],
        after: Sequence[Mapping[str, Any]],
        *,
        reason: str = "",
    ) -> Condensation | None:
        """Record that ``before`` became ``after``. Returns the condensation.

        Returns None when nothing was forgotten — a prune that changed nothing
        is not an event, and recording it would make the condensation list a
        turn counter.
        """
        with self._lock:
            self._observe_locked(before)
            before_ids = list(self._live)

            kept_ids, inserted = self._match_locked(before_ids, before, after)
            if kept_ids is None:
                self._breaks += 1
                logger.warning(
                    "Conversation log lost alignment during pruning (%d -> %d messages); "
                    "resynchronising rather than recording a condensation it cannot justify.",
                    len(before), len(after),
                )
                self._resync_locked(after)
                return None

            forgotten = [i for i in before_ids if i not in set(kept_ids)]
            if not forgotten and not inserted:
                self._live = kept_ids
                return None

            summary_id = 0
            summary_text = ""
            summary_offset = 0
            if inserted:
                summary_id = self._next_summary_id
                self._next_summary_id -= 1
                summary_text = "\n\n".join(_content_of(m) for _, m in inserted)
                summary_offset = inserted[0][0]
                # The summary is a real message in the live window, so it needs
                # an identity the log can hand back — not the synthetic event
                # View builds, which carries no role and none of the original
                # keys.
                self._messages[summary_id] = LoggedMessage(
                    event_id=summary_id, message=dict(inserted[0][1])
                )

            condensation = Condensation(
                forgotten_ids=tuple(forgotten),
                summary=summary_text,
                summary_offset=summary_offset,
                reason=reason or f"pruned {len(forgotten)} messages",
                tokens_reclaimed=sum(
                    estimate_tokens(self._messages[i].content)
                    for i in forgotten
                    if i in self._messages
                )
                - estimate_tokens(summary_text),
                summary_id=summary_id if inserted else -1,
            )

            self._condensations.append(condensation)
            self._watermarks.append(self._next_id - 1)
            self._trim_condensations_locked()
            self._live = self._window_ids_locked(kept_ids, condensation, summary_id)
            self._evict_locked()

            if not self._verify_locked(after):
                self._breaks += 1
                logger.warning(
                    "Conversation log derived a window that does not match the live "
                    "history; resynchronising."
                )
                self._resync_locked(after)
                return None

            return condensation

    def _window_ids_locked(
        self, kept_ids: list[int], condensation: Condensation, summary_id: int
    ) -> list[int]:
        """The live ids after applying this condensation, in window order."""
        if not condensation.summary:
            return list(kept_ids)
        without_summary = [i for i in kept_ids if i != summary_id]
        offset = min(condensation.summary_offset, len(without_summary))
        return [*without_summary[:offset], summary_id, *without_summary[offset:]]

    # -- alignment -------------------------------------------------------

    def _align_locked(
        self, known_ids: Sequence[int], history: Sequence[Mapping[str, Any]]
    ) -> tuple[list[int], int]:
        """Ids for ``history``, minting new ones for messages never seen."""
        result: list[int] = []
        cursor = 0
        minted = 0
        for message in history:
            matched = None
            for index in range(cursor, len(known_ids)):
                known = self._messages.get(known_ids[index])
                if known is not None and _same(known.message, message):
                    matched = known_ids[index]
                    cursor = index + 1
                    break
            if matched is None:
                matched = self._mint_locked(message)
                minted += 1
            result.append(matched)
        return result, minted

    def _mint_locked(self, message: Mapping[str, Any]) -> int:
        event_id = self._next_id
        self._next_id += 1
        self._messages[event_id] = LoggedMessage(event_id=event_id, message=dict(message))
        self._order.append(event_id)
        return event_id

    def _match_locked(
        self,
        before_ids: Sequence[int],
        before: Sequence[Mapping[str, Any]],
        after: Sequence[Mapping[str, Any]],
    ) -> tuple[list[int] | None, list[tuple[int, Mapping[str, Any]]]]:
        """Align ``after`` against ``before``.

        Returns ``(kept_ids, inserted)`` where ``inserted`` pairs each new
        message with its index in ``after``. Returns ``(None, [])`` when the
        result is not a subsequence-plus-insertions of the input, which means
        the pruner did something this log cannot describe and guessing would
        produce a record that reads as evidence while being wrong.
        """
        if len(before_ids) != len(before):
            return None, []

        kept: list[int] = []
        inserted: list[tuple[int, Mapping[str, Any]]] = []
        cursor = 0
        for position, message in enumerate(after):
            matched = None
            for index in range(cursor, len(before)):
                if _same(before[index], message):
                    matched = before_ids[index]
                    cursor = index + 1
                    break
            if matched is None:
                inserted.append((position, message))
            else:
                kept.append(matched)

        # More than one insertion is allowed — a strategy may add a preamble
        # beside its summary — but every insertion has to be contiguous at the
        # front of a span, or summary_offset cannot describe where they went.
        if inserted:
            positions = [p for p, _ in inserted]
            if positions != list(range(positions[0], positions[0] + len(positions))):
                return None, []
        return kept, inserted

    def _verify_locked(self, after: Sequence[Mapping[str, Any]]) -> bool:
        """Whether replaying the log reproduces the live window exactly.

        The guarantee this module offers is that the record describes the
        window, so it is checked rather than asserted in a docstring. A
        mismatch resynchronises and is counted.
        """
        derived = [self._messages[i].message for i in self._live if i in self._messages]
        if len(derived) != len(after):
            return False
        return all(_same(a, b) for a, b in zip(derived, after))

    def _resync_locked(self, history: Sequence[Mapping[str, Any]]) -> None:
        """Adopt the live history as-is after alignment failed."""
        self._live = []
        aligned, _ = self._align_locked([], history)
        self._live = aligned
        self._evict_locked()

    # -- bounds ----------------------------------------------------------

    def _evict_locked(self) -> None:
        """Drop the oldest messages once the log is full.

        Never a live one. A live message evicted from the log would leave the
        window unreconstructible from its own record, which is the single
        property this module exists to provide — so the bound is enforced
        against history, and a conversation whose live window alone exceeds the
        capacity simply stops accumulating history rather than corrupting it.
        """
        overflow = len(self._order) - self.capacity
        if overflow <= 0:
            return
        live = set(self._live)
        removed = 0
        remaining: list[int] = []
        for event_id in self._order:
            if removed < overflow and event_id not in live:
                self._messages.pop(event_id, None)
                removed += 1
                continue
            remaining.append(event_id)
        self._order = remaining
        self._evicted += removed

    def _trim_condensations_locked(self) -> None:
        excess = len(self._condensations) - self.condensation_capacity
        if excess > 0:
            del self._condensations[:excess]
            del self._watermarks[:excess]

    # -- reading ---------------------------------------------------------

    def live_history(self) -> list[dict[str, Any]]:
        """The window as it stands, derived from the log."""
        with self._lock:
            return [
                dict(self._messages[i].message) for i in self._live if i in self._messages
            ]

    def live_view(self) -> View:
        """The window as a condenser :class:`~core.context.condenser.View`."""
        with self._lock:
            events = tuple(
                self._messages[i].as_event() for i in self._live if i in self._messages
            )
            return View(events=events, condensations=tuple(self._condensations))

    def reconstruct(self, *, before_condensation: int) -> list[dict[str, Any]]:
        """The window as it stood just before the Nth condensation was applied.

        ``before_condensation=0`` is the history at the moment of the first
        forgetting. This is the question that could not be asked at all until
        now: what could she see when she answered that.

        Messages said *after* that condensation are excluded by its watermark,
        so the result is the window of the time rather than the current log with
        old forgettings replayed over it. Eviction bounds how far back this
        reaches; ``report()["evicted_from_log"]`` says whether it has bitten.
        """
        with self._lock:
            if before_condensation < 0:
                raise ValueError("before_condensation must be non-negative")
            applied = self._condensations[:before_condensation]
            if before_condensation < len(self._watermarks):
                watermark = self._watermarks[before_condensation]
            else:
                watermark = self._next_id - 1
            source = [
                self._messages[i].as_event()
                for i in self._order
                if i in self._messages and i <= watermark
            ]
            view = View.from_events(source, applied)
            out: list[dict[str, Any]] = []
            for event in view.events:
                logged = self._messages.get(event.event_id)
                out.append(
                    dict(logged.message)
                    if logged is not None
                    else {"role": "system", "content": event.content}
                )
            return out

    def condensations(self) -> list[Condensation]:
        with self._lock:
            return list(self._condensations)

    def report(self) -> dict[str, Any]:
        with self._lock:
            return {
                "messages_retained": len(self._order),
                "live_messages": len(self._live),
                "condensations": len(self._condensations),
                "messages_forgotten": sum(
                    len(c.forgotten_ids) for c in self._condensations
                ),
                "tokens_reclaimed": sum(
                    c.tokens_reclaimed for c in self._condensations
                ),
                "evicted_from_log": self._evicted,
                "alignment_breaks": self._breaks,
                "capacity": self.capacity,
            }


_log: ConversationLog | None = None
_log_lock = checked_lock("core.context.conversation_log")


def get_conversation_log() -> ConversationLog:
    """The process-wide log. Created on first use, never reset implicitly."""
    global _log
    if _log is None:
        with _log_lock:
            if _log is None:
                _log = ConversationLog()
    return _log
