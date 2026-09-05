"""core/voice/duplex/mind_bridge.py — Where the voice lane meets the mind.

The point of this whole package is that voice is not a separate assistant
bolted to the side of Aura. Every spoken turn goes through the same governed
CognitiveEngine path the desktop text surface uses, and every voice event —
interruption, backchannel, endpoint — is a real event on her bus rather than
a UI animation.

Two decisions here are load-bearing and worth stating plainly.

**We use the governed turn, not ``think_stream``.**
``CognitiveEngine.think_stream`` routes straight to the LLM router, skipping
the governance and validation phases that the desktop surface deliberately
requires — ``interface/server.py`` would rather fail closed than let a
surface show an ungoverned reply. Speaking an ungoverned answer out loud is
strictly worse than showing one, so the voice lane takes the same governed
path.

This used to cost the whole latency budget: the governed turn returned one
finished string, so time-to-first-audio was proportional to *total* reply
length and the filler reflex existed to paper over the wait. It no longer
does. ``respond_streaming`` watches the governed turn's own reply form —
the pipeline already produces it incrementally — and hands clauses out as
they land, each one governed before it is synthesised. The path, the checks
and the final answer are unchanged; only the waiting is gone. The finished
reply is still the authority, and a surface that spoke from the stream owes
the listener a reconciliation against it.

**Interruption edits what she believes she said.**
When the user cuts her off, she has "said" a paragraph that they only heard
the first clause of. If her memory keeps the whole thing, every later
reference to the unheard part is a hallucination from the user's side of the
conversation. So a barge-in records the spoken prefix and hands it to the
next turn as context.
"""
from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import re
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any

from core.runtime.errors import record_degradation
from core.utils.task_tracker import get_task_tracker

# How long the activity watcher waits for a telemetry event before
# re-checking whether it has been asked to stop. Silence is normal.
_ACTIVITY_POLL_TIMEOUT_S = 1.0

# How long to wait for a freshly created cognition task to actually enter the
# responder. This is one scheduling hop on any healthy loop; a budget this
# large means only a badly starved one ever reaches it.
_TURN_START_TIMEOUT_S = 5.0

logger = logging.getLogger("Aura.Voice.MindBridge")

# Voice-lane topics on the event bus. Namespaced so subscribers can take the
# whole lane with one prefix.
TOPIC_VOICE_EVENT = "voice.duplex"

Responder = Callable[..., Coroutine[Any, Any, str | None]]


@dataclass(slots=True)
class SpokenRecord:
    """What she actually got out loud, versus what she meant to say.

    ``spoken`` is the ground truth of the conversation: it is what the user
    heard. ``intended`` is only useful for diagnostics.
    """

    intended: str = ""
    spoken: str = ""
    interrupted: bool = False
    delivery_complete: bool = True
    started_at: float = 0.0
    ended_at: float = 0.0

    @property
    def unheard(self) -> str:
        """The tail she never delivered."""
        if not self.interrupted and self.delivery_complete:
            return ""
        intended = self.intended.strip()
        spoken = self.spoken.strip()
        if spoken and intended.startswith(spoken):
            return intended[len(spoken):].strip()
        return intended


async def _default_responder(
    transcript: str,
    *,
    effective_message: str,
    session_id: str,
    timeout_s: float,
    reply_stream: Any = None,
) -> str | None:
    """Refuse an unbound caller instead of bypassing governed chat admission."""
    record_degradation(
        "voice_duplex.mind",
        RuntimeError(
            "duplex voice responder is not bound to an authenticated chat surface"
        ),
        action="returned no reply without entering an ungoverned cognition lane",
    )
    return None


def _responder_accepts_reply_stream(responder: Responder) -> bool:
    """Whether this responder can deliver its reply incrementally.

    Asked once, at bind time. A responder that does not take the parameter is
    a perfectly good responder — it just answers all at once, and the surface
    falls back to speaking the finished reply.
    """
    try:
        signature = inspect.signature(responder)
    except (TypeError, ValueError):
        # Builtins and some callables have no introspectable signature. Assume
        # the narrower contract: a wrong guess here would raise TypeError on
        # every turn, which is a far worse failure than not streaming.
        return False
    for parameter in signature.parameters.values():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            return True
        if parameter.name == "reply_stream":
            return True
    return False


@dataclass(slots=True)
class StreamingTurn:
    """A governed turn in flight, plus the reply forming inside it.

    Two halves that must both be honoured. ``channel`` carries clauses as they
    are produced and is what makes speech start early; ``final()`` is the
    reply the turn stands behind and is the only text that is authoritative.
    Speaking from the first without checking the second is how a surface ends
    up having said something its own governance revised.
    """

    channel: Any
    task: "asyncio.Task[str | None]"

    async def final(self) -> str | None:
        """Await the finished governed reply. Safe to call more than once."""
        try:
            return await self.task
        except asyncio.CancelledError:
            raise
        except (RuntimeError, ValueError, AttributeError, TypeError, OSError) as exc:
            record_degradation(
                "voice_duplex.mind",
                exc,
                action="voice turn task failed after streaming had begun",
            )
            return None

    async def abandon(self) -> None:
        """Stop caring about this turn — used when a newer utterance wins."""
        self.channel.close()
        if not self.task.done():
            self.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.task


class MindBridge:
    """Cognition, telemetry and event-bus wiring for one voice session."""

    def __init__(
        self,
        *,
        session_id: str,
        responder: Responder | None = None,
        cognition_timeout_s: float = 120.0,
        spoken_reply_words: int = 45,
    ) -> None:
        self._session_id = session_id
        self._responder = responder or _default_responder
        self._timeout_s = cognition_timeout_s
        self._spoken_reply_words = int(spoken_reply_words)
        self._activity_task: asyncio.Task[None] | None = None
        self._activity_callback: Callable[[str], None] | None = None
        self._activity_stop = asyncio.Event()
        self._turn_active = False
        self._pending_interruption: SpokenRecord | None = None
        self._history: list[SpokenRecord] = []
        self._pending_corrections: list[str] = []
        self._responder_streams = _responder_accepts_reply_stream(self._responder)

    # ── cognition ────────────────────────────────────────────────────────

    async def respond(self, transcript: str, *, delivery_context: str = "") -> str | None:
        """One governed turn. Returns her reply text, or None if it failed."""
        transcript = (transcript or "").strip()
        if not transcript:
            return None

        effective = self._compose_effective_message(transcript, delivery_context)
        self._turn_active = True
        started = time.perf_counter()
        try:
            reply = await self._responder(
                transcript,
                effective_message=effective,
                session_id=self._session_id,
                timeout_s=self._timeout_s,
            )
        except TimeoutError:
            record_degradation(
                "voice_duplex.mind",
                TimeoutError(f"voice cognition exceeded {self._timeout_s}s"),
                action="told the user the reasoning lane timed out instead of inventing a reply",
            )
            return None
        except (RuntimeError, ValueError, AttributeError, ImportError, TypeError, OSError) as exc:
            record_degradation(
                "voice_duplex.mind",
                exc,
                action="voice turn failed before a reply formed; surfaced the failure",
            )
            return None
        finally:
            self._turn_active = False
            self._pending_interruption = None
            logger.info(
                "voice turn cognition: %.0f ms", (time.perf_counter() - started) * 1000.0
            )

        return (reply or "").strip() or None

    async def respond_streaming(
        self, transcript: str, *, delivery_context: str = ""
    ) -> "StreamingTurn | None":
        """One governed turn, with its reply readable while it forms.

        Same responder, same governed path, same final answer — the only
        difference is that the caller may watch the reply assemble instead of
        waiting for the last token. That is the whole latency story for
        speech: it is what lets the first clause be *said* while the rest is
        still being decided.

        Returns a handle rather than a string because the caller has two jobs,
        not one: consume the stream, then reconcile what it committed to out
        loud against the final governed reply. A caller that only wants the
        string should use ``respond``.
        """
        transcript = (transcript or "").strip()
        if not transcript:
            return None

        from core.conversation.reply_stream import ReplyStreamChannel

        effective = self._compose_effective_message(transcript, delivery_context)
        channel = ReplyStreamChannel()
        self._turn_active = True
        started = time.perf_counter()

        # A responder that cannot stream is not an error — it simply produces
        # no chunks, the caller sees an empty stream, and the finished reply is
        # spoken the ordinary way. Detecting that here rather than making every
        # binding accept a parameter it ignores keeps the responder contract
        # exactly as wide as it was.
        streaming_kwargs: dict[str, Any] = (
            {"reply_stream": channel} if self._responder_streams else {}
        )
        if not self._responder_streams:
            channel.close()

        entered = asyncio.Event()

        async def _run() -> str | None:
            entered.set()
            try:
                reply = await self._responder(
                    transcript,
                    effective_message=effective,
                    session_id=self._session_id,
                    timeout_s=self._timeout_s,
                    **streaming_kwargs,
                )
            except TimeoutError:
                record_degradation(
                    "voice_duplex.mind",
                    TimeoutError(f"voice cognition exceeded {self._timeout_s}s"),
                    action="told the user the reasoning lane timed out instead of inventing a reply",
                )
                return None
            except (
                RuntimeError, ValueError, AttributeError, ImportError, TypeError, OSError
            ) as exc:
                record_degradation(
                    "voice_duplex.mind",
                    exc,
                    action="voice turn failed before a reply formed; surfaced the failure",
                )
                return None
            finally:
                # The channel must close on every path, or a consumer that is
                # iterating it waits for a producer that has already gone.
                channel.close()
                self._turn_active = False
                self._pending_interruption = None
                logger.info(
                    "voice turn cognition: %.0f ms",
                    (time.perf_counter() - started) * 1000.0,
                )
            return (reply or "").strip() or None

        task = get_task_tracker().create_task(
            _run(), name=f"VoiceMindBridge.turn:{self._session_id}"
        )
        # Do not hand back a turn whose cognition has not started. Creating a
        # task only schedules it, and a "turn in flight" that has not actually
        # begun is a lie the caller then acts on: a replacement utterance
        # would cancel a turn that never ran, which from outside is
        # indistinguishable from the request being dropped on the floor.
        #
        # Bounded, because this is only ever one scheduling hop under any
        # healthy loop. If the loop is so wedged that a created task cannot
        # start within seconds, waiting longer helps nobody — the turn is
        # handed back anyway and the caller's own timeouts take it from there,
        # which is strictly better than a voice session that hangs at the
        # moment it is asked a question.
        try:
            await asyncio.wait_for(entered.wait(), timeout=_TURN_START_TIMEOUT_S)
        except TimeoutError:
            record_degradation(
                "voice_duplex.mind",
                TimeoutError(
                    f"cognition task did not start within {_TURN_START_TIMEOUT_S}s"
                ),
                action="handed back the turn anyway; the event loop is badly starved",
                severity="warning",
            )
        return StreamingTurn(channel=channel, task=task)

    def _reply_budget_for(self, transcript: str) -> tuple[int, bool]:
        """How long this answer should be. Returns (words, offer_to_continue).

        A single global word cap is a blunt instrument: it makes "what time
        is it?" verbose and "explain the tradeoff" truncated. Since reply
        length *is* time-to-first-audio on this path, matching length to the
        question is both a latency win and simply how people talk — you
        answer a yes/no question in a word and a hard question in a
        paragraph, then stop and see if they want more.
        """
        text = (transcript or "").strip().lower()
        if not text:
            return self._spoken_reply_words, False

        words = text.split()
        first = re.sub(r"[^\w']", "", words[0]) if words else ""

        # Explanatory: they asked for reasoning, so give room — but offer the
        # rest rather than delivering a lecture unprompted.
        if any(
            p in text
            for p in (
                "explain", "why do", "why does", "why is", "how does", "how do",
                "walk me through", "tell me about", "what's the difference",
                "compare", "trade-off", "tradeoff", "in detail", "elaborate",
            )
        ):
            return max(self._spoken_reply_words, 80), True

        # Factual/closed: a sentence is the honest length.
        if first in ("is", "are", "was", "were", "do", "does", "did", "can",
                     "could", "will", "would", "should", "have", "has", "did"):
            return 20, False
        if any(p in text for p in ("what time", "how many", "how much", "when is",
                                   "when's", "where is", "where's", "who is", "who's")):
            return 18, False

        return self._spoken_reply_words, False

    def _spoken_turn_directive(self, words: int, offer_more: bool) -> str:
        """Tell the engine this reply will be heard, not read.

        This used to carry two jobs, and one of them was really latency
        management in the costume of a style note: because the governed turn
        returned one finished string, reply length WAS time-to-first-audio, so
        the word cap was the only lever available. That is what made spoken
        answers shallower than the same question typed.

        Governed streaming removes that job (see governed_stream.py): clauses
        are released and spoken as they are produced, so the first word no
        longer waits for the last. What remains here is the real one — this is
        speech, and speech has a different shape from prose.

        The shape, not the size. A spoken turn is built from short clauses that
        each land on their own, because the listener has no way to re-read.
        Markdown is worse than long: a bullet list read aloud is a monotone run
        of fragments and a heading disappears entirely. Length is now a
        suggestion about density rather than a ceiling on substance.
        """
        tail = (
            " If there is more worth saying, say it — just keep it in the same "
            "spoken register rather than switching into an essay."
            if offer_more
            else " Lead with the useful part."
        )
        return (
            "[spoken turn: this is going to be heard, not read, in a live "
            "conversation. Talk the way a person talks — short clauses that "
            "each stand on their own, contractions, the natural connectives of "
            "speech. No markdown, no lists, no headings, no code blocks; say a "
            "list as a sentence. Around "
            f"{words} words is the usual shape, though say what the answer "
            f"actually needs.{tail}]"
        )

    def _compose_effective_message(
        self, transcript: str, delivery_context: str = ""
    ) -> str:
        """Build what the engine reasons over for this turn.

        The visible message stays the user's raw words; only this carries the
        machine annotations, so the transcript shown in the UI is never
        polluted with them.
        """
        words, offer_more = self._reply_budget_for(transcript)
        parts: list[str] = [self._spoken_turn_directive(words, offer_more)]

        # How they said it, when it stood out from how they usually sound.
        # Only notable deviations arrive here; a readout on every turn would
        # train her to ignore the channel entirely.
        if delivery_context:
            parts.append(delivery_context)

        # Anything she owes the listener from a previous turn — a reply that
        # her own governance revised after part of it was already heard. It is
        # handed over once and cleared, so she corrects the record and moves
        # on rather than apologising every turn thereafter.
        if self._pending_corrections:
            parts.extend(self._pending_corrections)
            self._pending_corrections = []

        pending = self._pending_interruption
        if pending is not None and (
            pending.interrupted or not pending.delivery_complete
        ):
            unheard = pending.unheard
            if unheard:
                heard = pending.spoken.strip()
                parts.append(
                    "[voice context: your previous spoken reply did not finish. "
                    f'The user heard only: "{heard[:400]}". '
                    f'They did not hear: "{unheard[:400]}". '
                    "Treat the unheard part as unsaid — do not assume they know it.]"
                )

        parts.append(transcript)
        return "\n\n".join(parts)

    # ── spoken-truth accounting ──────────────────────────────────────────

    def record_spoken(self, record: SpokenRecord) -> None:
        """Register what she actually delivered for this turn."""
        self._history.append(record)
        if len(self._history) > 32:
            self._history.pop(0)
        if record.interrupted or not record.delivery_complete:
            self._pending_interruption = record
            logger.info(
                "Interrupted after %d of %d chars; %d chars never heard",
                len(record.spoken),
                len(record.intended),
                len(record.unheard),
            )

    @property
    def last_spoken(self) -> SpokenRecord | None:
        return self._history[-1] if self._history else None

    def note_correction(self, context: str) -> None:
        """Carry a correction she owes the listener into her next turn.

        Used when something already spoken was revised by her own governance.
        The point is that the apology is *hers* — she is told what happened
        and words it herself on the next turn, rather than this module
        composing a fixed line about her own reliability.
        """
        text = str(context or "").strip()
        if text:
            self._pending_corrections.append(text[:1200])
            del self._pending_corrections[:-3]

    # ── telemetry: what is she actually doing right now ──────────────────

    async def start_activity_watch(self, callback: Callable[[str], None]) -> None:
        """Follow the engine's activity telemetry to keep fillers honest.

        Caveat, stated because it affects correctness: the telemetry topic is
        global, not per-session. If another surface were generating at the
        same instant, its activity could colour a filler here. The desktop WS
        handler already refuses concurrent turns, and the blast radius is
        limited to which *phrasing* of "one moment" she picks — never to the
        content of an answer.
        """
        if self._activity_task is not None:
            if not self._activity_task.done():
                return
            self._activity_task = None
        self._activity_callback = callback
        self._activity_stop.clear()
        self._activity_task = get_task_tracker().create_task(
            self._watch_activity(),
            name="VoiceMindBridge.activity_watch",
        )
        self._activity_task.add_done_callback(self._activity_watch_done)

    def _activity_watch_done(self, task: asyncio.Task[None]) -> None:
        if self._activity_task is task:
            self._activity_task = None
        if task.cancelled():
            return
        try:
            task.result()
        except (RuntimeError, AttributeError, TypeError, ValueError, ImportError) as exc:
            record_degradation(
                "voice_duplex.mind",
                exc,
                action="activity telemetry watcher exited; a later start can recover it",
                severity="warning",
            )

    async def _watch_activity(self) -> None:
        from core.event_bus import get_event_bus

        for attempt in range(4):
            queue: Any = None
            bus: Any = None
            try:
                bus = get_event_bus()
                queue = await bus.subscribe("telemetry")
                while not self._activity_stop.is_set():
                    # Bounded: an unbounded queue.get() cannot notice the stop
                    # event, so a quiet telemetry bus wedges this watcher for
                    # the life of the process and the task never retires. The
                    # timeout is a poll interval, not a failure — nothing
                    # arriving is the normal state between turns.
                    try:
                        item = await asyncio.wait_for(
                            queue.get(), timeout=_ACTIVITY_POLL_TIMEOUT_S
                        )
                    except TimeoutError:
                        continue
                    # The bus delivers (priority, seq, payload) tuples.
                    payload = (
                        item[2]
                        if isinstance(item, tuple) and len(item) >= 3
                        else item
                    )
                    if not isinstance(payload, dict) or not self._turn_active:
                        continue
                    if payload.get("type") != "activity":
                        continue
                    label = str(payload.get("label") or "")
                    key = self._activity_key_from_label(label)
                    if key and self._activity_callback:
                        self._activity_callback(key)
                return
            except asyncio.CancelledError:
                raise
            except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                record_degradation(
                    "voice_duplex.mind",
                    exc,
                    action=(
                        "activity telemetry watcher will retry"
                        if attempt < 3
                        else "activity telemetry watcher exhausted bounded retries"
                    ),
                    severity="warning",
                )
                if attempt < 3:
                    await asyncio.sleep(0.1 * (2**attempt))
            finally:
                if bus is not None and queue is not None:
                    with contextlib.suppress(Exception):
                        await bus.unsubscribe("telemetry", queue)

    @staticmethod
    def _activity_key_from_label(label: str) -> str:
        """Reverse the engine's human label back to an activity key."""
        low = label.lower()
        if "search" in low or "web" in low:
            return "sovereign_browser"
        if "terminal" in low or "command" in low:
            return "sovereign_terminal"
        if "network" in low:
            return "sovereign_network"
        if "file" in low:
            return "file_operation"
        if "image" in low or "render" in low:
            return "generate_image"
        if "memory" in low or "recall" in low:
            return "memory_recall"
        if "optimiz" in low or "evolv" in low or "code" in low:
            return "self_improvement"
        return ""

    async def stop_activity_watch(self) -> None:
        task = self._activity_task
        self._activity_task = None
        self._activity_callback = None
        self._activity_stop.set()
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    # ── event bus ────────────────────────────────────────────────────────

    async def publish(self, kind: str, payload: dict[str, Any] | None = None) -> None:
        """Emit a voice-lane event so the rest of her can see it happen."""
        try:
            from core.event_bus import get_event_bus

            await get_event_bus().publish(
                TOPIC_VOICE_EVENT,
                {
                    "kind": kind,
                    "session_id": self._session_id,
                    "timestamp": time.time(),
                    **(payload or {}),
                },
            )
        except (RuntimeError, AttributeError, TypeError, ValueError, ImportError) as exc:
            record_degradation(
                "voice_duplex.mind",
                exc,
                action=f"dropped voice event {kind!r}; the lane kept running",
                severity="debug",
            )

    def notify_user_spoke(self) -> None:
        """Tell the substrate voice engine a real user turn just landed.

        It uses this to reset follow-up timers, so without it she can decide
        to "organically follow up" on a conversation that is actively going.
        """
        try:
            from core.voice.substrate_voice_engine import get_substrate_voice_engine

            engine = get_substrate_voice_engine()
            if engine is not None:
                engine.on_user_spoke()
        except (ImportError, AttributeError, RuntimeError, TypeError) as exc:
            record_degradation(
                "voice_duplex.mind",
                exc,
                action="did not notify substrate voice engine of the user turn",
                severity="debug",
            )
