"""Context condensation as an event, not an edit.

Aura already had two ways to make a context fit: ``chat_compression`` truncates
a message list to a token budget, and ``context_compression`` degrades file
records through levels. Both mutate in place. Once either has run, the thing it
removed is gone from the structure and the only record of the decision is a log
line — you cannot ask "what did she stop being able to see, and when?", and you
cannot replay a turn as it actually looked.

This module makes forgetting a first-class, replayable event. The full log is
never edited. A ``Condensation`` records which events left the window, the
summary that stands in for them, and where it sits. A ``View`` is *derived*:
apply the condensations to the log and you get exactly the context that was live
at that moment — for this turn or for one three weeks ago.

Two consequences worth stating plainly:

* Condensation is deferred. A condenser returns the ``Condensation`` rather
  than a shortened list; the caller appends it and the next ``View.from_events``
  applies it. That is what keeps the log append-only and the history honest.
* **Pinned events are never forgotten.** A commitment sliding out of the context
  window is how an agent breaks a promise without ever deciding to. Aura keeps
  promises in the ledger, but the ledger cannot help if the turn that would act
  on one can no longer see it.

The strategy vocabulary (rolling window, amortized forgetting, LLM summarization,
observation masking, pipelines) is adapted from OpenHands' condenser design
(MIT-licensed — read for its semantics, written fresh here).
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import Callable, Iterable, Sequence

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.Condenser")

__all__ = [
    "ContextEvent",
    "Condensation",
    "View",
    "Condenser",
    "NoOpCondenser",
    "AmortizedForgettingCondenser",
    "LLMSummarizingCondenser",
    "ObservationMaskingCondenser",
    "PipelineCondenser",
    "estimate_tokens",
]


def estimate_tokens(text: str) -> int:
    """Cheap 4-chars-per-token estimate.

    Deliberately not a tokenizer call: this runs on every event of every view,
    and a wrong-by-10% budget is worth more than a correct one that costs a
    model round-trip.
    """
    return max(1, len(text) // 4)


@dataclass(frozen=True)
class ContextEvent:
    """One unit of conversation history.

    ``pinned`` marks an event as load-bearing — a commitment, a standing
    directive, a correction the user should not have to repeat. Condensers must
    route around it.
    """

    event_id: int
    kind: str
    content: str
    pinned: bool = False

    @property
    def tokens(self) -> int:
        return estimate_tokens(self.content)


@dataclass(frozen=True)
class Condensation:
    """The record of a forgetting.

    Append-only and self-describing: ``forgotten_ids`` plus ``summary_offset``
    is enough to rebuild the exact window without consulting anything else.
    """

    forgotten_ids: tuple[int, ...]
    summary: str
    summary_offset: int
    reason: str = ""
    tokens_reclaimed: int = 0
    #: Identity of the stand-in event this condensation inserts.
    #:
    #: Every summary used to be born as ``event_id=-1``, on the reasoning that a
    #: derived event should not be forgettable by a later condensation keyed on
    #: ids. The cost of that was steeper than the benefit. Two condensations
    #: produced two events with the same id, so a summary could not be referred
    #: to, could not be told apart from its predecessor, and — because
    #: ``_forgettable`` also skips ``kind == "condensation"`` — could never be
    #: subsumed. Summaries accumulated for the life of the conversation and the
    #: window filled with immortal ones.
    #:
    #: Distinct ids make re-summarization expressible: a later condensation can
    #: name an earlier summary among its ``forgotten_ids`` and fold it into a new
    #: one, which is what "integrate the previous snapshot" has always meant.
    #: Ids stay negative so they cannot collide with the source log.
    summary_id: int = -1

    def __post_init__(self) -> None:
        if self.summary_offset < 0:
            raise ValueError("summary_offset must be non-negative")
        if self.summary_id >= 0:
            raise ValueError(
                "summary_id must be negative so it cannot collide with a source event id"
            )

    @property
    def summary_event(self) -> ContextEvent:
        """The stand-in event inserted where the forgotten span used to be."""
        return ContextEvent(
            event_id=self.summary_id,
            kind="condensation",
            content=self.summary,
            pinned=False,
        )


@dataclass(frozen=True)
class View:
    """The context as the model actually sees it, derived from the log."""

    events: tuple[ContextEvent, ...] = ()
    condensations: tuple[Condensation, ...] = ()

    def __len__(self) -> int:
        return len(self.events)

    def __iter__(self):
        return iter(self.events)

    def __getitem__(self, index):
        return self.events[index]

    @property
    def tokens(self) -> int:
        return sum(event.tokens for event in self.events)

    @property
    def pinned_ids(self) -> frozenset[int]:
        return frozenset(e.event_id for e in self.events if e.pinned)

    @classmethod
    def from_events(
        cls,
        events: Iterable[ContextEvent],
        condensations: Iterable[Condensation] = (),
    ) -> View:
        """Replay the log through its condensations.

        This is the only way a View is built, which is what makes any past
        window reconstructible: keep the log and the condensations, get the
        context back.
        """
        events = tuple(events)
        condensations = tuple(condensations)

        # Sequentially, in the order they happened. Each condensation's offset
        # indexes the view *as it stood when that condensation was made*, so
        # they cannot be applied as one batched removal — the second one's
        # offset is meaningless against a list the first has not yet reshaped.
        kept = list(events)
        for condensation in condensations:
            forgotten = set(condensation.forgotten_ids)
            kept = [e for e in kept if e.event_id not in forgotten]
            if condensation.summary:
                offset = min(condensation.summary_offset, len(kept))
                kept.insert(offset, condensation.summary_event)

        return cls(events=tuple(kept), condensations=condensations)

    def apply(self, condensation: Condensation) -> View:
        """The view that results from also applying ``condensation``.

        Convenience for callers holding a View rather than the raw log; the
        canonical path remains from_events over the append-only log.
        """
        source = [e for e in self.events if e.kind != "condensation"]
        return View.from_events(source, (*self.condensations, condensation))


# ── condensers ─────────────────────────────────────────────────────────────


class Condenser(ABC):
    """A context-management strategy.

    ``condense`` returns the view unchanged when there is nothing to do, or a
    ``Condensation`` for the caller to append. It never returns a shortened
    view directly — that would edit history.
    """

    @abstractmethod
    def condense(self, view: View) -> View | Condensation:
        """Either the view as-is, or the forgetting that should be recorded."""

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"{type(self).__name__}()"


class NoOpCondenser(Condenser):
    """Never forgets. The honest default when a budget is not yet known."""

    def condense(self, view: View) -> View | Condensation:
        return view


class _RollingCondenser(Condenser):
    """Head / middle / tail split with a rolling window.

    Keeps ``keep_first`` events verbatim (the system framing and the original
    ask, which everything downstream is relative to) and as much of the recent
    tail as the target allows. The middle is what gets forgotten — plus never
    anything pinned, wherever it sits.
    """

    def __init__(
        self,
        max_size: int = 120,
        keep_first: int = 4,
        voice_anchors: int = 6,
        voice_kind: str = "assistant",
        subsume_summaries: bool = False,
    ) -> None:
        if max_size < 2:
            raise ValueError("max_size must be at least 2")
        if keep_first < 0:
            raise ValueError("keep_first must be non-negative")
        if keep_first >= max_size:
            raise ValueError(
                f"keep_first ({keep_first}) must be below max_size ({max_size}); "
                "otherwise the head alone overflows the window and nothing can "
                "ever be forgotten"
            )
        if voice_anchors < 0:
            raise ValueError("voice_anchors must be non-negative")
        self.max_size = max_size
        self.keep_first = keep_first
        self.voice_anchors = voice_anchors
        self.voice_kind = voice_kind
        #: Whether an earlier summary may be folded into a new one.
        #:
        #: Only safe when the summarizer is given the prior summaries and
        #: actually integrates them — which is why it is off by default and why
        #: :class:`AmortizedForgettingCondenser` refuses it outright: that
        #: strategy writes no summary at all, so subsuming one would delete the
        #: only remaining trace of the span it stood for.
        self.subsume_summaries = bool(subsume_summaries)

    def _voice_anchor_ids(self, view: View) -> frozenset[int]:
        """Her most recent turns, kept verbatim wherever they sit.

        Summarising is a register-destroying operation. A summary is written in
        summary voice — flat, third-person, informational — so replacing a span
        of her replies with a description of them removes the live examples of
        how she talks and leaves the model imitating its own meeting minutes.
        That is the mechanism behind "she stopped sounding like herself after a
        long conversation", and it is not hypothetical: the existing compressor
        flattens the oldest 70% of the thread into a <state_snapshot> once the
        window is half full.

        Commitments are pinned because losing them breaks a promise. These are
        anchored because losing them costs her the voice, which is the thing the
        conversation was for.
        """
        if self.voice_anchors <= 0:
            return frozenset()
        own = [e.event_id for e in view.events if e.kind == self.voice_kind]
        return frozenset(own[-self.voice_anchors:])

    @property
    def target_size(self) -> int:
        return self.max_size // 2

    def _should_condense(self, view: View) -> bool:
        return len(view) > self.max_size

    def _partition(
        self, view: View
    ) -> tuple[Sequence[ContextEvent], Sequence[ContextEvent], Sequence[ContextEvent]]:
        """Split into (head, middle, tail); middle is the forgetting candidate."""
        events = view.events
        head = events[: self.keep_first]
        tail_len = max(0, self.target_size - self.keep_first)
        tail = events[len(events) - tail_len :] if tail_len else ()
        middle = events[len(head) : len(events) - len(tail)]
        return head, middle, tail

    def _forgettable(
        self, middle: Sequence[ContextEvent], view: View | None = None
    ) -> list[ContextEvent]:
        """Middle events that may actually be dropped.

        Three things survive. Pinned events, because losing one breaks a
        promise. Prior summaries, because a summary carries the only remaining
        trace of what it replaced and forgetting it erases that span for good
        rather than compressing it. And voice anchors — her own recent turns —
        because a context with no verbatim examples of how she talks produces a
        reply that sounds like a summary of her.
        """
        anchors = self._voice_anchor_ids(view) if view is not None else frozenset()
        return [
            e for e in middle
            if not e.pinned
            and (self.subsume_summaries or e.kind != "condensation")
            and e.event_id not in anchors
        ]

    def _condensation(
        self, view: View, forgotten: Sequence[ContextEvent], summary: str, reason: str
    ) -> Condensation:
        return Condensation(
            forgotten_ids=tuple(e.event_id for e in forgotten),
            summary=summary,
            summary_offset=self.keep_first,
            reason=reason,
            tokens_reclaimed=sum(e.tokens for e in forgotten) - estimate_tokens(summary),
        )


class AmortizedForgettingCondenser(_RollingCondenser):
    """Drops the middle outright. No summary, no model call.

    The cheapest strategy and the right one when the middle is genuinely
    disposable — long tool-output runs, retry storms. It is lossy in the window
    but not in the log, which is the distinction the whole module rests on.
    """

    def __init__(self, *args, **kwargs) -> None:
        if kwargs.pop("subsume_summaries", False):
            raise ValueError(
                "AmortizedForgettingCondenser writes no summary, so subsuming an "
                "earlier one would drop the only remaining trace of the span it "
                "stood for"
            )
        super().__init__(*args, **kwargs)

    def condense(self, view: View) -> View | Condensation:
        if not self._should_condense(view):
            return view

        _, middle, _ = self._partition(view)
        forgotten = self._forgettable(middle, view)
        if not forgotten:
            # Everything in the middle was load-bearing. Refusing to forget is
            # the correct outcome; the caller's budget problem is real but the
            # answer is not to drop a commitment.
            logger.debug("nothing forgettable in the middle: all pinned")
            return view

        return self._condensation(
            view,
            forgotten,
            summary="",
            reason=f"amortized forgetting: dropped {len(forgotten)} events",
        )


class LLMSummarizingCondenser(_RollingCondenser):
    """Replaces the middle with a model-written summary.

    ``summarize`` takes the events being forgotten and returns prose. It is
    injected rather than reached for so this is testable without a cortex and
    so the caller chooses which model pays for it — a summary is exactly the
    kind of work that belongs on a small fast model.

    A failed summarization degrades to keeping the events. Forgetting them with
    no summary would silently lose the span, and a context that is too long is a
    smaller problem than one that is quietly wrong.
    """

    def __init__(
        self,
        summarize: Callable[[Sequence[ContextEvent]], str],
        max_size: int = 120,
        keep_first: int = 4,
        voice_anchors: int = 6,
        voice_kind: str = "assistant",
        subsume_summaries: bool = False,
    ) -> None:
        super().__init__(
            max_size=max_size,
            keep_first=keep_first,
            voice_anchors=voice_anchors,
            voice_kind=voice_kind,
            subsume_summaries=subsume_summaries,
        )
        self._summarize = summarize

    def condense(self, view: View) -> View | Condensation:
        if not self._should_condense(view):
            return view

        _, middle, _ = self._partition(view)
        forgotten = self._forgettable(middle, view)
        if not forgotten:
            return view

        try:
            summary = self._summarize(forgotten)
        except Exception as exc:  # noqa: BLE001 - degrade, never drop silently
            record_degradation(
                "context.condenser",
                exc,
                action="kept events unsummarized rather than forgetting them",
            )
            return view

        if not summary or not summary.strip():
            record_degradation(
                "context.condenser",
                ValueError("summarizer returned empty text"),
                action="kept events rather than forgetting them behind an empty summary",
            )
            return view

        return self._condensation(
            view,
            forgotten,
            summary=summary.strip(),
            reason=f"summarized {len(forgotten)} events",
        )


class ObservationMaskingCondenser(Condenser):
    """Blanks tool output older than ``attention_window`` events.

    Structure survives — the model still sees that it ran a command and in what
    order — while the bulk of stale stdout stops costing tokens. Distinct from
    forgetting: the events stay in the window and stay addressable.
    """

    PLACEHOLDER = "[tool output elided: outside the attention window]"

    def __init__(self, attention_window: int = 10, kinds: Iterable[str] = ("observation",)):
        if attention_window < 1:
            raise ValueError("attention_window must be at least 1")
        self.attention_window = attention_window
        self.kinds = frozenset(kinds)

    def condense(self, view: View) -> View | Condensation:
        events = view.events
        cutoff = len(events) - self.attention_window
        if cutoff <= 0:
            return view

        masked = []
        changed = False
        for index, event in enumerate(events):
            if (
                index < cutoff
                and event.kind in self.kinds
                and not event.pinned
                and event.content != self.PLACEHOLDER
            ):
                masked.append(replace(event, content=self.PLACEHOLDER))
                changed = True
            else:
                masked.append(event)

        if not changed:
            return view
        return View(events=tuple(masked), condensations=view.condensations)


class PipelineCondenser(Condenser):
    """Runs condensers in order, stopping at the first that wants to forget.

    Composition is the point: mask stale observations first, and only summarize
    if masking did not get the window under budget. Stopping at the first
    ``Condensation`` keeps the contract single-valued — the caller appends one
    event per step, so two forgettings in one step could not both be recorded.
    """

    def __init__(self, condensers: Iterable[Condenser]) -> None:
        self.condensers = tuple(condensers)
        if not self.condensers:
            raise ValueError(
                "a pipeline needs at least one condenser; use NoOpCondenser to "
                "express 'never forget' explicitly"
            )

    def condense(self, view: View) -> View | Condensation:
        current = view
        for condenser in self.condensers:
            result = condenser.condense(current)
            if isinstance(result, Condensation):
                return result
            current = result
        return current

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        inner = ", ".join(repr(c) for c in self.condensers)
        return f"PipelineCondenser({inner})"
