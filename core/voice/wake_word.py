"""core/voice/wake_word.py — Always-Listening Wake Word Detection
=================================================================
Runs as a background thread, minimal CPU. On detection of "Hey Aura",
raises foreground priority, starts a command session, and routes the
spoken command through the canonical conversation lane (/api/chat).

Voice is a surface, not a separate runtime: a spoken command enters the
exact same governed lane the desktop UI uses, so it gets identical
contracts, receipts, and capability dispatch.

Uses the existing Whisper transcript stream from the audio service.
Pattern-matches for wake phrases in transcript chunks.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from enum import StrEnum
from typing import Any

from core.container import ServiceContainer
from core.runtime.errors import record_degradation
from core.runtime.task_ownership import create_tracked_task
from core.voice.audio_provenance import attribute_wake_audio

logger = logging.getLogger("Aura.WakeWord")


class WakeState(StrEnum):
    IDLE = "idle"               # Passive listening for wake word
    LISTENING = "listening"     # Wake word detected, accumulating command
    PROCESSING = "processing"   # Command received, decomposing into task
    EXECUTING = "executing"     # Task graph running
    REPORTING = "reporting"     # Summarizing results


# Wake phrases (case-insensitive), written against what ASR ACTUALLY EMITS.
#
# LIVE DEFECT, 2026-08-10. "when i talk to my computer nothing happens. no
# response or anything." The microphone, the voice-activity gate, and the
# transcriber were all working perfectly; the log holds the proof:
#
#   Heard 'Hey, Aura, can you turn on your camera?' but did not answer:
#   no wake word and no open voice conversation.
#
# He said the wake word. It was transcribed correctly. `\bhey\s+aura\b` still
# missed, because Whisper punctuates and `\s+` cannot cross the comma in
# "Hey, Aura". Every utterance he spoke that day missed for the same reason,
# and the detector logged "OFFLINE (detected 0 wake events)" each session.
#
# The pattern had been tested against idealised strings — "hey aura" — which
# is the one form a real transcriber almost never produces for a sentence.
#
# The second half of the same failure: ASR renders her name as whatever it
# sounds like. The same log has "Hey, Laura, can you hear me right now?" from
# a man saying "Hey, Aura". A wake word that only matches the correct spelling
# of a name the transcriber routinely gets wrong is a wake word that does not
# work. These variants are what was observed, not a guess; each is anchored
# behind a greeting so an ordinary sentence cannot trip them.
#: Observed live in a single afternoon: "Aura", "Laura", "Orrick". A fixed list
#: of spellings loses to the next one the transcriber invents, so the name is
#: matched by shape instead — an optional leading consonant, an "or"/"au"/"ar"
#: vowel core, and whatever tail the transcriber appended. Anchored behind a
#: greeting, which is what keeps "laura called me yesterday" silent.
_WAKE_GAP = r"[\s,.\-–—:;]+"
_GREETING = r"(?:hey|hi|hello|ok(?:ay)?|yo)"
_NAME = r"(?:[lmnrd]?[oa]u?r[aeiouy]?\w{0,4})"
WAKE_PHRASES = [
    rf"\b{_GREETING}{_WAKE_GAP}{_NAME}\b",
    rf"\b{_NAME}\b.*\blisten\b",
]
WAKE_PATTERN = re.compile("|".join(WAKE_PHRASES), re.IGNORECASE)

# Interrupt phrases
INTERRUPT_PHRASES = [
    r"\bstop\b",
    r"\bcancel\b",
    r"\bpause\b",
    r"\bwait\b",
    r"\babort\b",
    r"\bnever\s*mind\b",
]
INTERRUPT_PATTERN = re.compile("|".join(INTERRUPT_PHRASES), re.IGNORECASE)


class WakeWordDetector:
    """Always-listening wake word detection.

    Lifecycle:
        IDLE → (wake word) → LISTENING → (VAD silence) → PROCESSING
        → EXECUTING → REPORTING → IDLE

    Barge-in: user can say "stop"/"cancel" at any point to interrupt.
    """

    SILENCE_TIMEOUT_S = 1.5      # silence after this = end of command
    SESSION_TIMEOUT_S = 30.0     # max session length
    POLL_INTERVAL_S = 0.2        # check transcript this often
    # Voice commands can trigger multi-step desktop chains; the lane call
    # is bounded so a wedged turn cannot strand the detector forever.
    COMMAND_TIMEOUT_S = float(os.environ.get("AURA_VOICE_COMMAND_TIMEOUT_S", "240") or 240)
    DESKTOP_COMMAND_TIMEOUT_S = float(
        os.environ.get("AURA_VOICE_DESKTOP_COMMAND_TIMEOUT_S", "660") or 660
    )
    SPEAK_TIMEOUT_S = 60.0
    SPOKEN_REPLY_CHAR_BUDGET = 600

    def __init__(self) -> None:
        self.state = WakeState.IDLE
        self._task: asyncio.Task | None = None
        self._dispatch_task: asyncio.Task | None = None
        self._session_start: float = 0.0
        self._last_speech: float = 0.0
        self._accumulated_transcript: str = ""
        self._last_processed_transcript: str = ""
        self._wake_count: int = 0
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        ServiceContainer.register_instance("wake_word", self, required=False)
        self._started = True
        self._task = create_tracked_task(
            self._detection_loop(),
            name="Aura.WakeWordDetector",
        )
        logger.info("WakeWordDetector ONLINE — listening for 'Hey Aura'")

    async def stop(self) -> None:
        self._started = False
        for task in (self._dispatch_task, self._task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        logger.info("WakeWordDetector OFFLINE (detected %d wake events)", self._wake_count)

    async def _detection_loop(self) -> None:
        """Main detection loop — reads from audio service transcript."""
        try:
            while self._started:
                try:
                    healer = ServiceContainer.get("self_healing", default=None)
                    if healer is not None:
                        healer.heartbeat("wake_word")
                    transcript = self._get_latest_transcript()

                    if self.state == WakeState.IDLE:
                        await self._check_wake_word(transcript)

                    elif self.state == WakeState.LISTENING:
                        await self._accumulate_command(transcript)

                    elif self.state in (WakeState.EXECUTING, WakeState.PROCESSING):
                        # Check for interrupts
                        if transcript and INTERRUPT_PATTERN.search(transcript):
                            await self._handle_interrupt()

                except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                    record_degradation("wake_word.loop", e)

                await asyncio.sleep(self.POLL_INTERVAL_S)

        except asyncio.CancelledError:
            raise

    def _get_latest_transcript(self) -> str:
        """Read the latest transcript from the audio service or WorldState."""
        try:
            ws = ServiceContainer.get("world_state", default=None)
            if ws and hasattr(ws, "last_voice_transcript"):
                transcript = ws.last_voice_transcript or ""
                if transcript != self._last_processed_transcript:
                    self._last_processed_transcript = transcript
                    return transcript
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation("wake_word.world_state_transcript", exc)

        # Try audio service directly
        try:
            import json
            from pathlib import Path

            audio_path = Path(__file__).resolve().parent.parent.parent / "sensory_audio.json"
            if audio_path.exists() and (time.time() - audio_path.stat().st_mtime) < 10:
                data = json.loads(audio_path.read_text(encoding="utf-8"))
                transcript = str(data.get("transcript") or data.get("text") or "")
                if transcript != self._last_processed_transcript:
                    self._last_processed_transcript = transcript
                    return transcript
        except (OSError, ValueError, TypeError) as exc:
            record_degradation("wake_word.audio_transcript_file", exc)

        return ""

    async def _verify_user_voice_print(self, transcript: str) -> dict[str, Any]:
        """Verify the speaker through a registered voice-identity service.

        Wake-word text is not identity proof. If no verifier is registered,
        return an explicit unverified result and continue under normal user
        governance without privilege escalation.
        """
        try:
            verifier = (
                ServiceContainer.get("voice_identity", default=None)
                or ServiceContainer.get("speaker_verifier", default=None)
            )
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("wake_word.voice_identity_lookup", exc)
            verifier = None
        if verifier is None:
            return {
                "verified": False,
                "confidence": 0.0,
                "reason": "voice_identity_verifier_unavailable",
            }

        try:
            verify = (
                getattr(verifier, "verify_current_speaker", None)
                or getattr(verifier, "verify_transcript", None)
                or getattr(verifier, "verify", None)
            )
            if not callable(verify):
                return {
                    "verified": False,
                    "confidence": 0.0,
                    "reason": "voice_identity_verifier_missing_verify_method",
                }
            result = verify(transcript)
            if hasattr(result, "__await__"):
                result = await result
            if isinstance(result, dict):
                verified = bool(result.get("verified"))
                confidence = float(result.get("confidence", 0.0) or 0.0)
                return {
                    "verified": verified,
                    "confidence": confidence,
                    "reason": str(result.get("reason") or "voice_identity_result"),
                    "verifier": type(verifier).__name__,
                }
            if isinstance(result, tuple):
                verified = bool(result[0])
                confidence = float(result[1]) if len(result) > 1 else (1.0 if verified else 0.0)
                return {
                    "verified": verified,
                    "confidence": confidence,
                    "reason": "voice_identity_tuple_result",
                    "verifier": type(verifier).__name__,
                }
            verified = bool(result)
            return {
                "verified": verified,
                "confidence": 1.0 if verified else 0.0,
                "reason": "voice_identity_bool_result",
                "verifier": type(verifier).__name__,
            }
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("wake_word.voice_identity_verify", exc)
            return {
                "verified": False,
                "confidence": 0.0,
                "reason": f"voice_identity_error:{type(exc).__name__}",
            }

    async def _check_wake_word(self, transcript: str) -> None:
        """Check for wake word in transcript."""
        if not transcript:
            return

        match = WAKE_PATTERN.search(transcript)
        if match:
            # Attribution BEFORE the session state is committed.
            #
            # OWNER REPORT, 2026-08-10: "videos playing on my computer are not
            # me speaking." Nothing here could tell the difference. Speaker
            # verification asks for a voice_identity service that this codebase
            # never registers — the only verify_current_speaker implementation
            # is a test double — so the verified branch below is unreachable in
            # production and every wake phrase, from any sound source in the
            # room, opened a command session as the owner.
            voice_evidence = await self._verify_user_voice_print(transcript)
            voice_evidence = attribute_wake_audio(voice_evidence)
            if not voice_evidence.get("owner_attributed", True):
                logger.info(
                    "🔇 Wake phrase heard while other audio is playing and the "
                    "speaker is unverified; not opening a command session (%s)",
                    voice_evidence.get("owner_attribution_reason", "unattributed"),
                )
                self._record_wake_observation(
                    "Wake phrase heard from unattributed audio — no session started",
                    voice_evidence,
                    salience=0.4,
                )
                return

            self._wake_count += 1
            self.state = WakeState.LISTENING
            self._session_start = time.time()
            self._last_speech = time.time()
            # Single-utterance support: "Hey Aura, open my notes" arrives as
            # one transcript chunk. The dedup in _get_latest_transcript means
            # this chunk will never be seen again, so the command portion must
            # be captured NOW or it is lost and the session times out empty.
            remainder = transcript[match.end():].lstrip(" ,.!?;:-—").strip()
            self._accumulated_transcript = remainder

            logger.info("🎤 Wake word detected! Starting command session #%d", self._wake_count)

            if voice_evidence.get("verified"):
                try:
                    from core.executive.authority_gateway import get_authority_gateway
                    gateway = get_authority_gateway()
                    token_id = gateway.issue_user_presence_token(
                        source="voice",
                        evidence=voice_evidence,
                    )
                    logger.info("Verified voice presence token issued: %s", token_id)
                except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
                    record_degradation("wake_word.user_presence_token", exc)
                    logger.error("Failed to issue user presence token: %s", exc)
            else:
                logger.info(
                    "Wake word accepted without verified speaker identity: %s",
                    voice_evidence.get("reason", "unverified"),
                )

            self._record_wake_observation(
                "Wake word detected — command session started",
                voice_evidence,
                salience=0.9,
            )

    def _record_wake_observation(
        self,
        summary: str,
        voice_evidence: dict[str, Any],
        *,
        salience: float,
    ) -> None:
        """Record what was heard AND how it was attributed.

        A refused wake is as much a fact about the room as an accepted one,
        and it is the only trace the owner would have that Aura heard the
        phrase and decided it was not him.
        """
        try:
            ws = ServiceContainer.get("world_state", default=None)
            if ws:
                ws.record_event(
                    summary,
                    source="voice",
                    salience=salience,
                    ttl=60,
                    metadata={"voice_identity": voice_evidence},
                )
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation("wake_word.world_state_event", exc)

    @staticmethod
    def _merge_transcript_chunk(existing: str, chunk: str) -> str:
        """Merge a transcript chunk into the accumulated command.

        The same utterance can arrive more than once through different
        ingestion paths (direct file read, then the perceptual pump's
        WorldState copy), possibly truncated differently. Re-deliveries
        must never REPLACE the accumulated command — that is how a long
        spoken objective got chopped to its tail mid-session. Substring
        re-deliveries are ignored; overlapping continuations are joined
        at the overlap; genuinely new speech is appended.
        """
        if not existing:
            return chunk
        if not chunk or chunk in existing:
            return existing
        max_k = min(len(existing), len(chunk))
        for k in range(max_k, 7, -1):
            if existing.endswith(chunk[:k]):
                return existing + chunk[k:]
        return f"{existing} {chunk}"

    async def _accumulate_command(self, transcript: str) -> None:
        """Accumulate spoken command after wake word."""
        now = time.time()

        if transcript:
            # Remove wake phrase from beginning
            command = WAKE_PATTERN.sub("", transcript).strip()
            if command:
                merged = self._merge_transcript_chunk(
                    self._accumulated_transcript, command
                )
                if merged != self._accumulated_transcript:
                    self._accumulated_transcript = merged
                    self._last_speech = now
                # Re-delivered chunks do not reset the silence window —
                # otherwise duplicate ingestion keeps the session open.

        # Check for end of command (silence timeout)
        silence_duration = now - self._last_speech
        session_duration = now - self._session_start

        if silence_duration > self.SILENCE_TIMEOUT_S and self._accumulated_transcript:
            # End of command — process it
            await self._process_command(self._accumulated_transcript)

        elif session_duration > self.SESSION_TIMEOUT_S:
            # Session timeout
            if self._accumulated_transcript:
                await self._process_command(self._accumulated_transcript)
            else:
                logger.info("Wake session timed out without command")
                self.state = WakeState.IDLE

    async def _process_command(self, command: str) -> None:
        """Hand a spoken command to the canonical conversation lane.

        Execution runs as a tracked background task so the detection loop
        keeps polling while the command executes — that is what keeps
        barge-in ("stop", "cancel") live during execution. The previous
        design awaited execution inline, which blocked the loop and made
        the interrupt branch unreachable for in-flight commands.
        """
        self.state = WakeState.PROCESSING
        self._accumulated_transcript = ""
        logger.info("🎤 Voice command received: '%s'", command[:100])

        if self._dispatch_task and not self._dispatch_task.done():
            logger.info("🎤 Superseding still-running voice command")
            self._dispatch_task.cancel()

        self.state = WakeState.EXECUTING
        self._dispatch_task = create_tracked_task(
            self._execute_command(command),
            name="Aura.VoiceCommandDispatch",
        )

    async def _execute_command(self, command: str) -> None:
        """Run one voice command through /api/chat and report the result."""
        try:
            ok, reply = await self._dispatch_to_conversation_lane(command)
            self.state = WakeState.REPORTING
            try:
                ws = ServiceContainer.get("world_state", default=None)
                if ws:
                    ws.record_event(
                        f"Voice command {'completed' if ok else 'failed'}: {reply[:160]}",
                        source="voice",
                        salience=0.8,
                        ttl=300,
                        command=command[:200],
                        ok=ok,
                    )
            except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                record_degradation(
                    "wake_word.command_event",
                    exc,
                    severity="warning",
                    action="completed the voice command while omitting only the optional world-state event",
                )
            if ok and reply:
                await self._speak_reply(reply)
            logger.info("🎤 Voice command %s", "completed" if ok else "failed")
        except asyncio.CancelledError:
            logger.info("🎤 Voice command cancelled mid-execution")
            raise
        finally:
            self._accumulated_transcript = ""
            self.state = WakeState.IDLE

    async def _dispatch_to_conversation_lane(self, command: str) -> tuple[bool, str]:
        """POST the command to the local /api/chat surface via the gateway.

        Loopback-only by construction; the transport hop runs under a local
        internal governed scope, and the governed effects (capability
        dispatch, desktop actions) are gated inside the lane itself.
        """
        from core.governance_context import local_internal_governed_scope
        from core.runtime.network_gateway import get_network_gateway

        try:
            port = int(os.environ.get("AURA_SERVER_PORT", "8000") or 8000)
        except ValueError:
            port = 8000
        timeout_s = self._conversation_lane_timeout(command)
        with local_internal_governed_scope(
            "wake_word.conversation_lane", domain="tool_execution"
        ):
            response = await get_network_gateway().request_async(
                "POST",
                f"http://127.0.0.1:{port}/api/chat",
                headers={
                    "Content-Type": "application/json",
                    "X-Aura-Surface": "voice",
                },
                data=json.dumps(
                    {"message": command, "session_id": "voice-wake"}
                ).encode("utf-8"),
                timeout=timeout_s,
                source="wake_word:conversation_lane",
            )
        if not response.get("ok") or int(response.get("status_code") or 0) != 200:
            detail = str(response.get("error") or response.get("status_code") or "unknown")
            record_degradation(
                "wake_word.conversation_lane",
                RuntimeError(f"conversation lane dispatch failed: {detail}"),
            )
            return False, f"conversation lane dispatch failed: {detail}"
        try:
            payload = json.loads(response.get("content") or b"{}")
        except (ValueError, TypeError) as exc:
            record_degradation("wake_word.conversation_lane_payload", exc)
            return False, "conversation lane returned an unreadable payload"
        reply = str(payload.get("response") or "").strip()
        return bool(reply), reply

    def _conversation_lane_timeout(self, command: str) -> float:
        """Choose a bounded wait budget for the voice->chat request.

        Normal conversation stays snappy, but explicit desktop objectives can
        legitimately keep the HTTP request open while governed tools execute
        and receipts are collected. A short voice timeout makes the neural
        stream report "Voice command failed" even though the action later
        succeeds, so desktop objectives get the same long-running budget as
        the live proof path.
        """
        base = max(1.0, float(self.COMMAND_TIMEOUT_S or 240.0))
        try:
            from core.runtime.desktop_objective_intent import looks_like_desktop_objective

            if looks_like_desktop_objective(command):
                return max(base, float(self.DESKTOP_COMMAND_TIMEOUT_S or 660.0))
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
            return base
        return base

    async def _speak_reply(self, reply: str) -> None:
        """Speak a bounded portion of the reply if a voice engine is live."""
        spoken = reply[: self.SPOKEN_REPLY_CHAR_BUDGET]
        if len(reply) > self.SPOKEN_REPLY_CHAR_BUDGET:
            cut = spoken.rfind(". ")
            if cut > 200:
                spoken = spoken[: cut + 1]
        try:
            voice = ServiceContainer.get("voice_engine", default=None)
            speak = getattr(voice, "speak", None) if voice is not None else None
            if not callable(speak):
                return
            result = speak(spoken)
            if hasattr(result, "__await__"):
                await asyncio.wait_for(result, timeout=self.SPEAK_TIMEOUT_S)
        except TimeoutError as exc:
            record_degradation("wake_word.speak_timeout", exc)
        except (AttributeError, RuntimeError, TypeError, ValueError, OSError) as exc:
            record_degradation("wake_word.speak", exc)

    async def _handle_interrupt(self) -> None:
        """Handle a spoken interrupt ("stop", "cancel", etc.)."""
        logger.info("🎤 Voice interrupt received — cancelling current action")
        if self._dispatch_task and not self._dispatch_task.done():
            self._dispatch_task.cancel()
        self.state = WakeState.IDLE
        self._accumulated_transcript = ""

    def get_status(self) -> dict[str, Any]:
        return {
            "running": bool(self._started and self._task and not self._task.done()),
            "state": self.state.value,
            "wake_count": self._wake_count,
            "session_active": self.state != WakeState.IDLE,
            "accumulated": self._accumulated_transcript[:60] if self._accumulated_transcript else "",
        }


_instance: WakeWordDetector | None = None


def get_wake_word_detector() -> WakeWordDetector:
    global _instance
    if _instance is None:
        _instance = WakeWordDetector()
    return _instance


__all__ = ["WakeWordDetector", "WakeState", "get_wake_word_detector"]
