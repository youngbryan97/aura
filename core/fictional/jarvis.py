"""JARVIS — ProactiveAnticipationEngine.

Does not wait to be asked: watches the host, notices open threads, and
initiates. The difference between a tool and a partner.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from typing import Any

from core.fictional.common import (
    disk_percent_value,
    record_fictional_degradation,
)
from core.runtime import resource_psutil as psutil
from core.runtime.lockdep import LockRank, checked_lock
from core.runtime.service_access import resolve_orchestrator

logger = logging.getLogger("Aura.FictionalSynthesis")


class FictionalEngine:
    """Shim for legacy/orphaned references to FictionalEngine."""
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.created_at = time.time()

    def get_status(self) -> dict[str, Any]:
        return {
            "status": "shim_active",
            "created_at": self.created_at,
            "args": len(self.args),
            "kwargs": sorted(self.kwargs.keys()),
        }

# ═══════════════════════════════════════════════════════════════════════════════
# ENGINE 1: JARVIS — ProactiveAnticipationEngine
# ═══════════════════════════════════════════════════════════════════════════════

class ProactiveAnticipationEngine:
    """
    Derived from: J.A.R.V.I.S. (Iron Man)
    
    JARVIS's most underappreciated property: he notices things and brings
    them up. He doesn't wait. When the reactor power drops 3%, he says so.
    When Pepper is calling while Tony is in the lab, he routes it intelligently.
    When there's a pattern in the news that relates to Tony's current work,
    he flags it.
    
    This engine monitors Aura's environment continuously and fires proactive
    initiations when conditions are met. Not intrusive — rate limited and
    context-aware. But genuinely anticipatory.
    """

    MIN_INITIATION_INTERVAL_S = 300    # Don't initiate more than once per 5 min
    MAX_DAILY_INITIATIONS = 20         # Daily cap to prevent annoyance
    WATCH_DIRS: list[str] = []         # Directories to watch for file changes

    # CP126: "JARVIS state is unsynchronized and unresolved topics are
    # unbounded. Concurrent callers mutate rate and topic structures without
    # a lock, and unresolved topics can grow without retention or size
    # limits."
    #
    # Both were true. self._lock was constructed and never acquired, and a
    # topic whose reminder had fired was never removed — so the list only
    # grew, and _check_unresolved_topics rescanned every dead entry on every
    # cycle for the life of the process.
    #
    # The retention window is derived from what the reminder actually says.
    # It fires after 24h with the words "you mentioned something yesterday";
    # past 48h that sentence is false, so the entry can no longer produce a
    # truthful reminder and is dead weight rather than pending work.
    UNRESOLVED_TOPIC_RETENTION_S = 48 * 3600
    MAX_UNRESOLVED_TOPICS = 200
    MAX_INTEREST_KEYWORDS = 50

    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator
        self._last_initiation_time: float = 0.0
        self._daily_initiation_count: int = 0
        self._daily_reset_date: str = ""
        self._running = False
        self._pending_initiations: asyncio.Queue = None
        self._system_baseline: dict[str, float] = {}
        #: Last announced magnitude per standing condition, so a fact that
        #: stays true is stated once rather than every cycle it holds.
        self._condition_last_value: dict[str, float | None] = {}
        self._unresolved_topics: deque = deque(maxlen=self.MAX_UNRESOLVED_TOPICS)
        self._user_interest_keywords: deque = deque(maxlen=self.MAX_INTEREST_KEYWORDS)
        self._last_user_activity: float = time.time()
        self._conversation_patterns: deque = deque(maxlen=100)
        # Ranked so lockdep can see it. A lock this module constructs and
        # never acquires — which is what was here — is indistinguishable
        # from no lock at all, except that it reads as protection.
        self._lock = checked_lock("jarvis.proactive_state", rank=LockRank.LEAF)
        self._cycle_failure_count = 0
        logger.info("ProactiveAnticipationEngine initialized (JARVIS pattern)")

    def _reset_daily_count_if_needed(self):
        """Caller must hold ``self._lock``.

        Read-then-write on the date and the counter: two turns landing on
        the stroke of midnight could both see a stale date and both reset,
        letting the day's initiation budget be spent twice.
        """
        today = time.strftime("%Y-%m-%d")
        if today != self._daily_reset_date:
            self._daily_initiation_count = 0
            self._daily_reset_date = today

    #: Phrases that suggest the SPEAKER left something open. Whose text
    #: they appear in decides whose commitment it is, which is the part
    #: that was missing: the marker was matched against Aura's reply and
    #: the topic was then filed under the user's message, so Aura saying
    #: "I'll look into that" produced "you mentioned something yesterday
    #: that we didn't finish" a day later (CP126 ``36b05560``).
    _OPEN_THREAD_MARKERS = (
        "later", "tomorrow", "we should", "next time", "to be continued",
        "let me think", "i'll",
    )
    #: The user asking to be reminded is the one case where a marker in
    #: the USER's text names a commitment Aura owes back to them.
    _REMINDER_REQUEST_MARKERS = ("remind me", "don't forget", "dont forget")

    def record_activity(self, user_input: str = "", response: str = ""):
        """Call after every conversation turn.

        An open thread is recorded with WHO left it open and on what basis.
        Neither is a receipt of an agreed commitment, so both travel with
        the entry and the reminder says what it is rather than asserting a
        promise that may never have been made.
        """
        opened = self._infer_open_thread(user_input, response)
        now = time.time()
        with self._lock:
            self._last_user_activity = now
            self._reset_daily_count_if_needed()

            if user_input:
                self._conversation_patterns.append({
                    "input": user_input[:200],
                    "timestamp": now,
                    "resolved": False,
                })

            if opened is not None:
                author, basis, topic = opened
                self._unresolved_topics.append({
                    "topic": topic[:200],
                    "author": author,
                    "basis": basis,
                    "inferred": True,
                    "timestamp": now,
                    "reminder_fired": False,
                })
                self._prune_unresolved_topics(now)

    @classmethod
    def _infer_open_thread(
        cls, user_input: str, response: str
    ) -> tuple[str, str, str] | None:
        """Return (author, basis, topic) when a turn left something open.

        Returns None rather than guessing. There is no commitment receipt
        in this codebase yet, so this is an inference and is labelled one
        everywhere it is used.
        """
        user_lower = (user_input or "").lower()
        response_lower = (response or "").lower()

        for marker in cls._REMINDER_REQUEST_MARKERS:
            if marker in user_lower:
                return "user_request", marker, user_input
        for marker in cls._OPEN_THREAD_MARKERS:
            if marker in user_lower:
                return "user", marker, user_input
        for marker in cls._OPEN_THREAD_MARKERS:
            if marker in response_lower:
                # Aura's own words. The thread is HERS, and the topic is
                # what she said, not what was asked.
                return "aura", marker, response
        return None

    def _prune_unresolved_topics(self, now: float) -> None:
        """Caller must hold ``self._lock``. Drop spent and expired entries."""
        live = [
            topic for topic in self._unresolved_topics
            if not topic.get("reminder_fired")
            and (now - float(topic.get("timestamp", 0.0))) <= self.UNRESOLVED_TOPIC_RETENTION_S
        ]
        if len(live) != len(self._unresolved_topics):
            self._unresolved_topics.clear()
            self._unresolved_topics.extend(live)

    def record_interest(self, keywords: list[str]):
        """Call when user demonstrates interest in topics."""
        with self._lock:
            existing = set(self._user_interest_keywords)
            for kw in keywords:
                if kw not in existing:
                    # deque(maxlen=…) evicts its own oldest; the previous
                    # rebinding slice was a race, since a concurrent reader
                    # kept a list that had silently stopped being the state.
                    self._user_interest_keywords.append(kw)
                    existing.add(kw)

    @staticmethod
    def _sample_host_blocking() -> dict[str, float | None]:
        """Read host metrics. Syscalls; never call this on the loop."""
        cpu = _a_cpu_reading()
        mem = psutil.virtual_memory()
        from core.runtime.disk_budget import state_volume_percent

        return {
            "cpu_percent": cpu,
            "memory_percent": mem.percent,
            "memory_available_gb": mem.available / (1024**3),
            "disk_percent": disk_percent_value(state_volume_percent),
        }

    async def _sample_system_state(self) -> dict[str, float]:
        """Get current system metrics, off the event loop.

        ``virtual_memory`` and the state-volume read are statvfs/sysctl
        calls. They are individually short and this runs every two minutes
        forever, which is exactly the shape of latency that never shows up
        in a profile and is always present.
        """
        try:
            return await asyncio.to_thread(self._sample_host_blocking)
        except (ImportError, OSError, AttributeError, RuntimeError) as e:
            record_fictional_degradation(
                e,
                action="skipped proactive system anomaly detection after host metric sampling failed",
            )
            logger.debug("System sampling failed: %s", e)
            return {}

    def _can_initiate(self) -> bool:
        """Check rate limits before firing an initiation.

        Held under the lock end to end. Checking the budget and spending it
        in separate critical sections is the classic way a "max 20 per day"
        cap delivers 21 — two concurrent callers both read 19.
        """
        with self._lock:
            return self._can_initiate_locked()

    def _can_initiate_locked(self) -> bool:
        """Caller must hold ``self._lock``."""
        self._reset_daily_count_if_needed()
        now = time.time()
        idle_seconds = now - self._last_user_activity

        if idle_seconds < 30:
            return False  # User is active — don't interrupt
        if now - self._last_initiation_time < self.MIN_INITIATION_INTERVAL_S:
            return False
        if self._daily_initiation_count >= self.MAX_DAILY_INITIATIONS:
            return False
        return True

    #: The lines past which the host is worth her attention: a name, the figure read, and the line in percent.
    _HOST_LINES = (("cpu", "cpu_percent", 90.0), ("memory", "memory_percent", 85.0), ("disk", "disk_percent", 90.0))

    #: The least time her turn on a reading may take; a longer prompt gets what it needs on this machine. The bound
    #: her remark on the screen holds to (core/perception/ambient_utterance.py).
    COMPOSE_LEAST_S = 25.0

    #: How much a continuing condition must move before it is worth saying again.
    #:
    #: Percentage points, because every condition guarded this way is measured
    #: in them. Below this, a repeat carries no information the person does not
    #: already have.
    CONDITION_RESTATEMENT_DELTA = 5.0

    def _condition_is_worth_restating(self, key: str, value: float | None) -> bool:
        """True when a keyed condition is new, materially changed, or re-armed.

        LIVE DEFECT, 2026-08-18. "Disk is 92% full. Worth cleaning up before it
        causes problems." had been fired 436 times — 265 at 92%, 109 at 93%, 54
        at 94%, 8 at 99%. Each reading was true when taken, so nothing was
        wrong with the measurement; the same standing fact was simply
        re-announced every cycle it remained true.

        The daily budget did not prevent it, because the budget counts
        initiations and not SUBJECTS: one unchanging condition spent the whole
        allowance, and everything else she might have raised was crowded out by
        it. A cap on how often she speaks is not a cap on how often she repeats
        herself.

        Keyed on the condition rather than the sentence, so rewording does not
        restart the nagging, and re-armed when the condition clears — the point
        is to say a thing once while it is true, not to say it once ever.
        """
        if not key:
            return True
        with self._lock:
            previous = self._condition_last_value.get(key)
            if previous is None:
                self._condition_last_value[key] = value
                return True
            if value is None or previous is None:
                return False
            if abs(float(value) - float(previous)) < self.CONDITION_RESTATEMENT_DELTA:
                return False
            self._condition_last_value[key] = value
            return True

    def note_condition_cleared(self, key: str) -> None:
        """The condition no longer holds, so saying it again would be news."""
        if not key:
            return
        with self._lock:
            self._condition_last_value.pop(key, None)

    async def _fire_initiation(
        self,
        reading: str,
        priority: str = "low",
        *,
        heading: str = "WHAT SHE NOTICED",
        condition: str = "",
        value: float | None = None,
    ) -> bool:
        """Hand her what was noticed, and spend the budget only if words of hers reach the person.

        It sent its own sentences as hers, with a label in front. LIVE 15 November 2026, 16:11, in her conversation:
        "[Proactive/JARVIS] CPU usage is at 100% — something is running hot. Want me to check what's consuming
        resources?" (F06, canned answers out). What was noticed is a reading now, under ``heading``, beside her recent
        conversation on her cortex (core/brain/her_unprompted_turn.py); the sentence, if any, is hers, and her Will
        decides after. When nothing of hers forms, nothing is said.

        The daily cap and the interval clock used to advance the moment the
        attempt began. A dropped emit, a missing output path or a raised
        exception therefore consumed a slot out of "at most N a day", and
        the initiations that WOULD have reached the person were suppressed
        by the ones that never did (CP126 ``d1252922``).

        The budget is reserved before the await, because two cycles can
        overlap, and released when nothing was delivered.
        """
        # Checked BEFORE the budget: a repeat of something already said must
        # not consume a slot, which is the whole failure being fixed.
        if not self._condition_is_worth_restating(condition, value):
            return False
        if not self._reserve_initiation():
            return False

        logger.info("JARVIS noticed: %s", " ".join(reading.split())[:80])

        delivered = False
        try:
            from core.container import ServiceContainer

            # The bus is a broadcast, not a delivery. Emitting to it does
            # not mean a person saw anything, so it does not count.
            bus = ServiceContainer.get("mycelium", default=None)
            if bus:
                await bus.emit("aura.proactive.initiation", {
                    "content": f"{heading}: {reading}",
                    "priority": priority,
                    "source": "jarvis_anticipation",
                    "timestamp": time.time(),
                })

            orch = self.orchestrator or resolve_orchestrator()
            if orch is not None and hasattr(orch, "emit_spontaneous_message"):
                from core.brain.her_unprompted_turn import her_turn_on

                said = await her_turn_on(heading, reading, origin="jarvis", least_s=self.COMPOSE_LEAST_S)
                if said:
                    await orch.emit_spontaneous_message(said, origin="jarvis")
                    delivered = True
                else:
                    logger.info("JARVIS: nothing of hers formed on %s", heading.lower())
            elif orch is not None and getattr(orch, "reasoning_queue", None):
                orch.reasoning_queue.put_nowait({
                    "text": f"## {heading}\n{reading}",
                    "is_proactive": True,
                    "priority": priority,
                    "source": "jarvis",
                })
                delivered = True
            else:
                logger.warning(
                    "JARVIS: No output path (reply/reasoning queue) for initiation: %s",
                    heading,
                )
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError, asyncio.QueueFull) as e:
            record_fictional_degradation(
                e,
                action="dropped proactive initiation after event bus and output routing failed",
            )
            logger.error("JARVIS: Initiation emit failed: %s", e)
        finally:
            if not delivered:
                self._release_initiation()
        return delivered

    def _reserve_initiation(self) -> bool:
        """Take a slot out of the daily budget, under the lock."""
        with self._lock:
            if not self._can_initiate_locked():
                return False
            self._last_initiation_time = time.time()
            self._daily_initiation_count += 1
            return True

    def _release_initiation(self) -> None:
        """Give the slot back. Nothing reached the person."""
        with self._lock:
            self._daily_initiation_count = max(0, self._daily_initiation_count - 1)
            self._last_initiation_time = 0.0

    async def _check_system_anomalies(self):
        """JARVIS-style environmental awareness — flag hardware issues."""
        state = await self._sample_system_state()
        if not state:
            return

        if not self._system_baseline:
            self._system_baseline = state
            return

        # The lines the host is past, raised as one reading and once while they hold. An unread figure is past none.
        crossed = {name: float(state[field]) for name, field, line in self._HOST_LINES if (state.get(field) or 0) > line}
        if not crossed:
            # The condition no longer holds, so the next time it does is news. Nothing called this before, so a
            # condition that cleared stayed said until its figure moved 5 points.
            with self._lock:
                said = [name for name in self._condition_last_value if name.startswith("host:")]
            for name in said:
                self.note_condition_cleared(name)
            return
        await self._fire_initiation(
            _the_host_as_read(state), priority="high" if "disk" in crossed else "medium", heading="THE HOST, READ NOW",
            condition="host:" + "+".join(sorted(crossed)), value=max(crossed.values()))

    async def _check_unresolved_topics(self):
        """Remind user of things they said they'd return to.

        Claims one topic under the lock, then awaits outside it. Iterating
        the live structure across an await let record_activity() append
        mid-iteration; and marking the topic fired only after the await
        meant a second cycle entering during the await picked the same
        topic and reminded twice.
        """
        now = time.time()
        claimed = None
        with self._lock:
            self._prune_unresolved_topics(now)
            for topic in self._unresolved_topics:
                if topic.get("reminder_fired"):
                    continue
                if (now - float(topic.get("timestamp", 0.0))) / 3600 > 24:
                    # Claim it before releasing the lock, not after the await.
                    topic["reminder_fired"] = True
                    claimed = topic
                    break
        if claimed is None:
            return
        await self._fire_initiation(self._the_thread_as_read(claimed, now), priority="low", heading="A THREAD LEFT OPEN")
        with self._lock:
            self._prune_unresolved_topics(time.time())

    @staticmethod
    def _the_thread_as_read(topic: dict[str, Any], now: float) -> str:
        """Whose thread it was, what was said, how long ago, and what that rests on.

        One sentence for all three authors put Aura's own unfinished sentence in the user's mouth, and asserted a
        commitment nobody made. The reading says which of the three it is, and an inference says it is one.
        """
        excerpt = str(topic.get("topic", ""))[:80]
        author = str(topic.get("author", "user"))
        hours = max(0.0, (now - float(topic.get("timestamp", now) or now)) / 3600.0)
        whose = {
            "aura": "her own words, a thread she said she would come back to",
            "user_request": "the person's words, asking to be reminded",
        }.get(author, "the person's words, taken as unfinished from how they were phrased; an inference, not a request")
        return f'"{excerpt}"\n{whose}\n{hours:.0f} hours ago, and not taken up since'

    async def _check_pending_agency_goals(self):
        """Surface goals the agency engine has queued but not acted on."""
        try:
            from core.container import ServiceContainer
            agency = ServiceContainer.get("agency_core", default=None)
            if not agency:
                return
            ctx = agency.get_emotional_context() if hasattr(agency, 'get_emotional_context') else {}
            pending = ctx.get("pending_goals", 0)
            if pending > 3:
                await self._fire_initiation(
                    f"{pending} of her goals queued and not yet acted on", priority="low",
                    heading="HER GOALS NOT YET WORKED ON",
                )
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as e:
            record_fictional_degradation(
                e,
                action="continued proactive cycle without pending agency-goal prompt",
            )
            logger.debug("JARVIS: Agency goal check failed: %s", e)

    async def run_cycle(self):
        """Single monitoring cycle — call from heartbeat loop."""
        if not self._can_initiate():
            return

        await self._check_system_anomalies()
        await self._check_unresolved_topics()
        await self._check_pending_agency_goals()

    async def start(self, interval_seconds: float = 120.0):
        """Run continuously in background."""
        if self._running:
            return
        self._running = True
        logger.info("ProactiveAnticipationEngine running (%.0fs intervals)", interval_seconds)
        while self._running:
            try:
                await self.run_cycle()
                self._cycle_failure_count = 0
            except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                self._cycle_failure_count += 1
                record_fictional_degradation(
                    e,
                    action=(
                        "backed off proactive anticipation loop after cycle failure "
                        f"#{self._cycle_failure_count}"
                    ),
                )
                logger.error("Anticipation cycle error: %s", e)
            await asyncio.sleep(min(interval_seconds * max(1, self._cycle_failure_count), 600.0))

    def stop(self):
        self._running = False



def _the_host_as_read(state: dict[str, float | None]) -> str:
    """The machine she runs on as read this cycle: each figure that was read, and none that was not."""
    figures = (
        ("processor in use", state.get("cpu_percent"), "{:.0f}%"),
        ("memory in use", state.get("memory_percent"), "{:.0f}%"),
        ("memory available", state.get("memory_available_gb"), "{:.1f} GB"),
        ("data volume in use", state.get("disk_percent"), "{:.0f}%"),
    )
    return "\n".join(f"{name}: {shown.format(float(value))}" for name, value, shown in figures if value is not None)


def _a_cpu_reading() -> float | None:
    """The host's processor use, or None when there is no reading to give.

    Read through the psutil shim, an unavailable or not-yet-taken figure came back as 100%. LIVE 15 November 2026,
    16:11: "CPU usage is at 100% — something is running hot" in her conversation, in her voice, five seconds after her
    health pulse read 17.6%.
    """
    from core.runtime.resource_observation import get_resource_observer

    compute = get_resource_observer().compute()
    return float(compute.cpu_percent) if compute.cpu_percent_readable else None
