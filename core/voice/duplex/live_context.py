"""Her voice mode's live context: one conversation her model reads once, keeps, and only ever adds to.

Each turn used to rebuild her prompt and her model read all of it before her first word: LIVE 11 November 2026, "What's
two plus two?" in voice mode read 2,354 tokens (10.9 s) to say "Four." Here a voice session keeps exactly the text her
model has read and said. When the person begins to speak, the turn's evidence is read into it (the read-ahead, which
runs while they talk); when they stop, only their words and the opening of her answer are new. On her cortex the first
token came 0.32 s after the words against 5.4 to 6.7 s with the prompt rebuilt (tools/voice/prove_a_live_context.py),
and the text tokenized back to exactly what she had read and generated on every turn, which is what lets her worker's
cache find it again.

The text is ChatML as her model was trained on it, with no system prompt (her identity is in her weights, F06): the
conversation so far, each turn's evidence at the head of the person's turn, then their words. Her reply goes in as she
produced it. A turn that asks her to act, or about what she can do or her runtime, goes to her governed chat route
instead, where her tools and instruments are; so does any turn when the context is not ready.
"""
from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import time
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

logger = logging.getLogger("Aura.Voice.LiveContext")

_ANSWER_OPENS = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
_TURN_ENDS = "<|im_end|>\n"


def rendered(exchanges: Sequence[dict[str, Any]]) -> str:
    """The conversation so far as her model reads it."""
    turns = []
    for exchange in exchanges:
        said, answered = str(exchange.get("user") or "").strip(), str(exchange.get("aura") or "").strip()
        if said and answered:
            turns.append(f"<|im_start|>user\n{said}{_TURN_ENDS}<|im_start|>assistant\n{answered}{_TURN_ENDS}")
    return "".join(turns)


def what_she_reads_first(exchanges: Sequence[dict[str, Any]], now: float | None = None) -> list[dict[str, Any]]:
    """The conversation a voice session opens on: the sitting going on now while it is short enough to be read whole,
    and always the last exchange, which a new turn answers into.

    It is what her governed turns read of history before the request bears on anything (history_reach.measure_reach):
    earlier sittings reach her as memory, and what is read whole is bounded by A_SITTING_READ_WHOLE_CHARS. LIVE 11
    November 2026, opening on the forty most recent exchanges whole read 12,326 tokens and held her worker for the
    first minute of the session; her first spoken turn waited behind it and took 69 s.
    """
    from core.conversation.delivered_history import _attested_pairs
    from core.conversation.history_reach import (
        A_SITTING_READ_WHOLE_CHARS,
        _exchange_chars,
        _this_sitting,
    )

    pairs = _attested_pairs(exchanges)
    if not pairs:
        return []
    sitting = list(_this_sitting(pairs, now) or [])
    if sum(_exchange_chars(pairs[index]) for index in sitting) > A_SITTING_READ_WHOLE_CHARS:
        sitting = []
    return [pairs[index] for index in sorted({*sitting, len(pairs) - 1})]


def the_turns_evidence() -> str:
    """What is so at the moment the person begins: read while they speak, so it must not depend on their words."""
    try:
        from core.brain.present_moment import present_moment_block

        present = present_moment_block().strip()
    except (ImportError, RuntimeError, ValueError, OSError) as exc:
        logger.debug("the present moment could not be read for this turn: %s", exc)
        present = ""
    if not present:
        return ""
    return f"[GROUNDING EVIDENCE FOR THIS TURN]\n{present}\n[END GROUNDING EVIDENCE FOR THIS TURN]\n\n"


def goes_to_her_governed_route(transcript: str) -> bool:
    """A turn that asks her to act, or about what she can do or her runtime, needs her tools and instruments."""
    try:
        from core.runtime.self_state_intent import (
            asked_to_act_in_a_capability_domain,
            asks_about_own_capabilities,
            asks_about_own_runtime,
        )
    except ImportError:
        return True
    return bool(asked_to_act_in_a_capability_domain(transcript) or asks_about_own_capabilities(transcript)
                or asks_about_own_runtime(transcript))


class HerLiveContext:
    """The text her model has read and said in this voice session, and the turn being read ahead."""

    def __init__(self, session_id: str, *, read: Callable[[str, str], Awaitable[dict[str, Any] | None]] | None = None,
                 generate: Callable[..., Awaitable[str | None]] | None = None,
                 record: Callable[[str, str], Awaitable[None]] | None = None) -> None:
        self.scope = f"voice_live:{str(session_id or 'voice')[:40]}"
        self.text = ""
        self.ready = False
        self.opening = False
        self._opened = asyncio.Event()
        self._opening = ""
        self._reading: asyncio.Task | None = None
        self._speaking = False  # a person began a turn that has not been answered yet
        self._said_into = ""  # the text up to where her last reply begins, until her next turn
        self._governed = ""  # the words of a turn her governed route is answering now
        self._read = read or _read_ahead_on_her_cortex
        self._generate = generate or _generate_on_her_cortex
        self._record = record

    async def open(self, exchanges: Sequence[dict[str, Any]] | Awaitable[Sequence[dict[str, Any]]]) -> None:
        """Read the conversation so far, once. A turn that ends while this is under way waits for it; one that
        comes after it failed takes her governed route."""
        self.opening = True
        try:
            if inspect.isawaitable(exchanges):
                exchanges = await exchanges
            self.text = rendered(what_she_reads_first(exchanges))
            if self.text:
                held = await self._read(self.text, self.scope)
                if not held or held.get("status") != "ok":
                    logger.info("🗣️ her live context could not be read (%s); voice turns take her governed route", held)
                    return
            self.ready = True
        finally:
            self.opening = False
            self._opened.set()
        logger.info("🗣️ her live context is open: %d characters of this conversation read once", len(self.text))
        if self._speaking:
            # They began before it opened (LIVE 11 November 2026, 0.45 s before): their turn's evidence is read now.
            self.a_person_began_to_speak()

    def a_person_began_to_speak(self) -> None:
        """Read the turn's evidence while they talk."""
        self._speaking = True
        if not self.ready:
            return
        self._opening = f"<|im_start|>user\n{the_turns_evidence()}"
        if self._reading is not None and not self._reading.done():
            return  # a read of this turn is already under way
        from core.utils.task_tracker import get_task_tracker

        self._reading = get_task_tracker().create_task(self._read(self.text + self._opening, self.scope),
                                                       name=f"VoiceLiveContext.read_ahead:{self.scope}")

    def answers(self, transcript: str) -> bool:
        return ((self.ready or self.opening) and bool((transcript or "").strip())
                and not goes_to_her_governed_route(transcript))

    async def answer(self, transcript: str, *, reply_stream: Any = None, timeout_s: float = 90.0) -> str | None:
        """Her answer on the live context, streamed into ``reply_stream`` as it forms; None when it did not form."""
        words = " ".join(str(transcript or "").split())
        if not self.ready:
            # LIVE 11 November 2026: the person stopped while it was opening, and the governed route took 16 s.
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._opened.wait(), timeout=timeout_s)
            if not self.ready:
                return None
        if self._reading is not None:
            try:
                await asyncio.wait_for(asyncio.shield(self._reading), timeout=timeout_s)
            except (TimeoutError, asyncio.CancelledError, RuntimeError):
                logger.debug("the read-ahead did not finish; the turn reads what is left itself")
        opening = self._opening or f"<|im_start|>user\n{the_turns_evidence()}"
        prompt = self.text + opening + words + _ANSWER_OPENS
        self._said_into, self._governed = prompt, ""
        try:
            reply = await self._generate(prompt, scope=self.scope, words=words, reply_stream=reply_stream,
                                         timeout_s=timeout_s)
        finally:
            self._opening, self._reading, self._speaking = "", None, False
        if not reply or not reply.strip():
            self._said_into = ""
            return None
        # What she said goes in as she said it, so the next turn's prompt begins with exactly what her model holds.
        self.text = prompt + reply + _TURN_ENDS
        if self._record is not None:
            await self._record(words, reply.strip())
        return reply.strip()


    def heard_elsewhere(self, words: str, reply: str) -> None:
        """A turn her governed route answered goes into the context too; the next read-ahead reads it."""
        said, answered = " ".join(str(words or "").split()), str(reply or "").strip()
        self._governed = ""
        if self.ready and said and answered:
            self._said_into = self.text + f"<|im_start|>user\n{said}{_TURN_ENDS}<|im_start|>assistant\n"
            self.text = self._said_into + answered + _TURN_ENDS
        self._opening, self._reading, self._speaking = "", None, False

    def cut_off(self, said: str) -> None:
        """She was cut off: her last reply goes into the conversation as far as she got out loud, whether it was still
        being written or already whole. LIVE 11 November 2026, a reply cut off while it was written never went in,
        and "Wait, stop. Just tell me in one sentence." reached her with nothing before it, and she
        answered "I am an AI assistant here to help you with any questions or tasks you have."
        """
        said = str(said or "").strip()
        if self._said_into:
            self.text, self._said_into = self._said_into + said + _TURN_ENDS, ""
        elif self._governed and self.ready and said:  # her governed route was answering; its reply never returned
            self.text += rendered([{"user": self._governed, "aura": said}])
        self._governed = ""

    def governed_turn(self, words: str) -> None:
        """Her governed route takes this turn: if she is cut off before its reply returns, the turn still goes in."""
        self._governed, self._said_into = " ".join(str(words or "").split()), ""


def _her_cortex_client() -> Any:
    from core.runtime.service_access import resolve_inference_gate

    return getattr(resolve_inference_gate(default=None), "_mlx_client", None)


async def _read_ahead_on_her_cortex(text: str, scope: str) -> dict[str, Any] | None:
    """Read while the person speaks, as a person's turn: background work on her model stops for it now."""
    from core.brain.llm.her_live_context import read_ahead
    from core.brain.llm_health_router import (
        _a_person_waits,
        _lease_is_user_foreground,
        _oldest_generation_gate_lease,
    )

    with _a_person_waits():
        holder = _oldest_generation_gate_lease()
        if holder is not None and not _lease_is_user_foreground(holder[0], holder[2]):
            from core.brain.llm.mlx_client import soft_cancel_active_generations

            soft_cancel_active_generations(reason="voice_live:a person began to speak")
        return await asyncio.to_thread(read_ahead, _her_cortex_client(), text, scope)


async def _generate_on_her_cortex(prompt: str, *, scope: str, words: str, reply_stream: Any, timeout_s: float) -> str | None:
    """Her answer from her cortex on the raw live prompt, as a person's turn: background work stands aside for it."""
    from core.brain.llm_health_router import (
        _a_person_waits,
        acquire_external_generation_gate_lease,
        release_external_generation_gate_lease,
    )
    from core.brain.llm_health_router_gate_owners import _background_stands_aside
    from core.conversation.reply_stream import bind_reply_stream

    client = _her_cortex_client()
    if client is None or not hasattr(client, "generate"):
        return None
    lease = None
    deadline = time.monotonic() + max(0.0, float(timeout_s))
    with _a_person_waits():
        # What of her background work holds her model is asked to stop now, and again each second she waits: one
        # that took the gate after the turn began was never asked. LIVE 11 November 2026, her narrative on her CPU
        # model took it while its worker was still starting, the turn's one ask found nothing to cancel, and her
        # reply waited 84 s behind it.
        while lease is None and time.monotonic() < deadline:
            _background_stands_aside("voice_live: a person is waiting for her reply")
            lease = await acquire_external_generation_gate_lease(
                owner="voice_live:reply", timeout_s=timeout_s,
                wait_s=min(1.0, max(0.0, deadline - time.monotonic())), foreground=True)
    if lease is None:
        return None
    hold = pacing = None
    try:
        if reply_stream is not None and hasattr(reply_stream, "backlog"):
            from core.brain.llm.speech_hold import SpeechHold

            hold = SpeechHold()
            paced = {"holds": 0, "held_s": 0.0, "most_waiting": 0}
            pacing = asyncio.ensure_future(_her_thought_waits_for_her_voice(hold, reply_stream, paced))
        # The contracts a governed reply carries; with them her worker sends each checked sentence as it forms, and
        # her voice speaks it while the rest is still being made (core/brain/llm/reply_previews.py).
        kwargs = dict(foreground_request=True, stream_user_surface=reply_stream is not None, origin="voice_live",
                      purpose="reply", prompt_cache_scope=scope, clean_user_surface_contract=True,
                      semantic_completion_contract=True, user_surface_validation_prompt=words, max_tokens=1024,
                      timeout=timeout_s, speech_hold=hold.name if hold is not None else "")
        if reply_stream is None:
            return await _her_reply(client, prompt, kwargs, timeout_s)
        with bind_reply_stream(reply_stream):
            return await _her_reply(client, prompt, kwargs, timeout_s)
    finally:
        if pacing is not None:
            pacing.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await pacing
            logger.info("🗣️ her thought waited for her voice %d time(s), %.1f s; at most %d waiting to be spoken",
                        paced["holds"], paced["held_s"], paced["most_waiting"])
        if hold is not None:
            hold.close()
        release_external_generation_gate_lease(lease)


async def _her_reply(client: Any, prompt: str, kwargs: dict[str, Any], timeout_s: float) -> str | None:
    """Her reply, its completion owned by its progress as her governed turns' is: held for her voice, a long reply
    outlasts any clock set before it began (inference_gate's foreground path does the same)."""
    complete = getattr(client, "generate_text_to_completion", None)
    if not callable(complete):
        return await client.generate(prompt, **kwargs)
    from core.utils.deadlines import Deadline

    return await complete(prompt=prompt, deadline=Deadline(timeout=timeout_s), **kwargs)


async def _her_thought_waits_for_her_voice(hold: Any, reply_stream: Any, paced: dict[str, Any] | None = None) -> None:
    """While a sentence of hers waits to be spoken, her model holds; when her voice takes it, her model goes on. The
    hold lapses by itself a second after this stops saying so. ``paced`` counts it for her log."""
    paced = paced if paced is not None else {"holds": 0, "held_s": 0.0, "most_waiting": 0}
    holding = False
    while True:
        waiting = reply_stream.backlog()
        paced["most_waiting"] = max(paced["most_waiting"], waiting)
        if waiting > 0:
            hold.hold_for(1.0)
            paced["holds"] += not holding
            paced["held_s"] += 0.1
        else:
            hold.release()
        holding = waiting > 0
        await asyncio.sleep(0.1)
