"""core/bus/qos.py — quality-of-service profiles for the event bus.

Clean-room adoption of the DDS/ROS 2 QoS model, layered over Aura's
existing `AuraEventBus` rather than replacing it.

An event bus with one delivery policy forces every topic into the same
trade-off, and the trade-offs genuinely differ:

* A **sensor stream** should be best-effort with history depth 1. The
  newest sample is the only one that matters; queuing stale frames behind
  a slow consumer makes the consumer see the past.
* A **state announcement** — "the model lane is warm", "autonomy is
  paused" — must be *transient-local*: a subscriber that starts after the
  announcement still needs to know. This is the single biggest gap in a
  volatile-only bus, and it produces exactly the bug Aura has hit: an
  organ boots after a state event and behaves as though the state never
  changed, forever, because nothing will republish it.
* A **command** must be reliable and must not silently vanish under
  backpressure.
* A **heartbeat** wants a *deadline*: not receiving it is the signal.
  Without deadline QoS, absence is invisible — you can only notice a
  message you received.

The other half of the ROS 2 model, and the half people forget, is **QoS
compatibility**. A subscriber requesting stronger guarantees than the
publisher offers does not silently degrade — in DDS it simply never
connects, which is the classic afternoon-losing ROS bug. Here the
mismatch is *reported*, loudly, at subscribe time, and the subscription is
still made with the offered policy. An incompatibility you can see beats
both silent degradation and silent disconnection.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
import time
from collections import deque
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field, replace
from enum import IntEnum, StrEnum
from typing import Any

logger = logging.getLogger("Aura.QoS")


class Reliability(IntEnum):
    """Ordered by strength — a request may not exceed the offer."""

    BEST_EFFORT = 0
    RELIABLE = 1


class Durability(IntEnum):
    """VOLATILE: late joiners get nothing. TRANSIENT_LOCAL: they get the
    retained history, which is what makes state announcements work."""

    VOLATILE = 0
    TRANSIENT_LOCAL = 1


#: How long a stream with no declared deadline waits before waking to
#: re-check whether it should still be running. A subscriber that can only
#: be woken by a publisher cannot notice its own shutdown, and on a quiet
#: topic that is indistinguishable from a wedge.
STREAM_POLL_S = 1.0


class History(StrEnum):
    KEEP_LAST = "keep_last"
    KEEP_ALL = "keep_all"


@dataclass(frozen=True)
class QosProfile:
    reliability: Reliability = Reliability.RELIABLE
    durability: Durability = Durability.VOLATILE
    history: History = History.KEEP_LAST
    depth: int = 10
    #: Samples older than this are never delivered. 0 disables.
    lifespan_s: float = 0.0
    #: Expected maximum gap between samples. 0 disables.
    deadline_s: float = 0.0
    #: A publisher that has not asserted within this is declared not alive.
    liveliness_lease_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "reliability": self.reliability.name.lower(),
            "durability": self.durability.name.lower(),
            "history": str(self.history),
            "depth": self.depth,
            "lifespan_s": self.lifespan_s,
            "deadline_s": self.deadline_s,
            "liveliness_lease_s": self.liveliness_lease_s,
        }

    def incompatibilities(self, offered: QosProfile) -> list[str]:
        """What this profile requests that ``offered`` does not provide."""
        problems: list[str] = []
        if self.reliability > offered.reliability:
            problems.append(
                f"requested {self.reliability.name} but publisher offers "
                f"{offered.reliability.name}: delivery may be dropped"
            )
        if self.durability > offered.durability:
            problems.append(
                f"requested {self.durability.name} but publisher offers "
                f"{offered.durability.name}: nothing is retained for late joiners"
            )
        if self.deadline_s and offered.deadline_s and offered.deadline_s > self.deadline_s:
            problems.append(
                f"requested a {self.deadline_s:.2f}s deadline but publisher only "
                f"guarantees {offered.deadline_s:.2f}s"
            )
        if (
            self.liveliness_lease_s
            and offered.liveliness_lease_s
            and offered.liveliness_lease_s > self.liveliness_lease_s
        ):
            problems.append(
                f"requested a {self.liveliness_lease_s:.2f}s liveliness lease but "
                f"publisher asserts every {offered.liveliness_lease_s:.2f}s"
            )
        return problems


#: Named profiles for the four shapes above, so callers pick an intent
#: rather than assembling five fields and getting one wrong.
SENSOR_DATA = QosProfile(
    reliability=Reliability.BEST_EFFORT,
    durability=Durability.VOLATILE,
    depth=1,
    lifespan_s=2.0,
)
STATE = QosProfile(
    reliability=Reliability.RELIABLE,
    durability=Durability.TRANSIENT_LOCAL,
    depth=1,
)
COMMAND = QosProfile(
    reliability=Reliability.RELIABLE,
    durability=Durability.VOLATILE,
    history=History.KEEP_ALL,
    depth=256,
)
HEARTBEAT = QosProfile(
    reliability=Reliability.BEST_EFFORT,
    durability=Durability.VOLATILE,
    depth=1,
    deadline_s=10.0,
    liveliness_lease_s=15.0,
)
DEFAULT = QosProfile()


@dataclass
class Sample:
    topic: str
    data: Any
    published_at: float
    sequence: int

    def expired(self, lifespan_s: float, now: float | None = None) -> bool:
        if lifespan_s <= 0:
            return False
        return ((now or time.time()) - self.published_at) > lifespan_s


@dataclass
class TopicState:
    """Everything the QoS layer keeps per topic."""

    profile: QosProfile
    retained: deque[Sample] = field(default_factory=lambda: deque(maxlen=1))
    sequence: int = 0
    last_published_at: float = 0.0
    last_assert_at: float = 0.0
    published: int = 0
    dropped: int = 0
    deadline_misses: int = 0
    liveliness_losses: int = 0
    alive: bool = True


class QosBus:
    """QoS semantics over the underlying event bus."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._topics: dict[str, TopicState] = {}
        self._mismatches: list[dict[str, Any]] = []
        self._deadline_callbacks: dict[str, list[Callable[[str, float], None]]] = {}
        self._liveliness_callbacks: dict[str, list[Callable[[str], None]]] = {}

    # ── declaration ───────────────────────────────────────────────────
    def declare_publisher(self, topic: str, profile: QosProfile = DEFAULT) -> TopicState:
        """Declare the QoS a topic is published with. Idempotent per topic."""
        with self._lock:
            state = self._topics.get(topic)
            if state is not None:
                if state.profile != profile:
                    logger.warning(
                        "topic %r re-declared with a different QoS profile; keeping "
                        "the first (%s) — a topic has one contract",
                        topic,
                        state.profile.to_dict(),
                    )
                return state
            maxlen = None if profile.history is History.KEEP_ALL else max(1, profile.depth)
            state = TopicState(profile=profile, retained=deque(maxlen=maxlen))
            self._topics[topic] = state
            return state

    def offered(self, topic: str) -> QosProfile:
        with self._lock:
            state = self._topics.get(topic)
            return state.profile if state else DEFAULT

    def check_compatibility(self, topic: str, requested: QosProfile) -> list[str]:
        """Report, do not silently degrade and do not silently disconnect."""
        offered = self.offered(topic)
        problems = requested.incompatibilities(offered)
        if problems:
            entry = {
                "topic": topic,
                "requested": requested.to_dict(),
                "offered": offered.to_dict(),
                "problems": problems,
                "at": time.time(),
            }
            with self._lock:
                self._mismatches.append(entry)
                if len(self._mismatches) > 64:
                    del self._mismatches[:-64]
            logger.warning(
                "QoS mismatch on %r: %s", topic, "; ".join(problems)
            )
        return problems

    # ── publishing ────────────────────────────────────────────────────
    async def publish(
        self, topic: str, data: Any, *, profile: QosProfile | None = None, priority: int | None = None
    ) -> bool:
        state = self.declare_publisher(topic, profile) if profile else self._state(topic)
        now = time.time()
        with self._lock:
            state.sequence += 1
            if state.profile.durability is Durability.TRANSIENT_LOCAL:
                # Retain a snapshot, never the caller's object. The bus
                # stamps routing metadata onto dict payloads in place, and
                # publishers reuse buffers; a retained sample that aliases
                # either one hands late joiners something that has changed
                # since it was announced.
                state.retained.append(
                    Sample(
                        topic=topic,
                        data=dict(data) if isinstance(data, dict) else data,
                        published_at=now,
                        sequence=state.sequence,
                    )
                )
            gap = now - state.last_published_at if state.last_published_at else 0.0
            state.last_published_at = now
            state.last_assert_at = now
            state.published += 1
            deadline = state.profile.deadline_s
            missed = bool(deadline and state.last_published_at and gap > deadline)
            if missed:
                state.deadline_misses += 1
            if not state.alive:
                state.alive = True

        if missed:
            self._fire_deadline(topic, gap)

        try:
            from core.event_bus import EventPriority, get_event_bus

            bus = get_event_bus()
            await bus.publish(
                topic,
                data,
                priority=priority if priority is not None else EventPriority.COGNITIVE,
            )
            return True
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                state.dropped += 1
            if state.profile.reliability is Reliability.RELIABLE:
                logger.error("reliable publish to %r failed: %s", topic, exc)
                from core.runtime.errors import record_degradation

                with contextlib.suppress(Exception):
                    record_degradation(
                        "qos_bus",
                        exc,
                        severity="degraded",
                        action=f"reliable sample on {topic} was lost",
                        enforce_failure_policy=False,
                    )
            else:
                logger.debug("best-effort publish to %r dropped: %s", topic, exc)
            return False

    def assert_liveliness(self, topic: str) -> None:
        """A publisher with nothing to say still says it is there."""
        with self._lock:
            self._state(topic).last_assert_at = time.time()

    # ── subscribing ───────────────────────────────────────────────────
    async def subscribe(
        self, topic: str, *, profile: QosProfile = DEFAULT
    ) -> tuple[asyncio.Queue, list[Sample]]:
        """Subscribe with a requested profile.

        Returns the live queue and any retained samples the subscriber is
        entitled to under transient-local durability — delivered as a
        return value rather than pushed, so the caller handles history and
        live samples in one obvious order.
        """
        self.check_compatibility(topic, profile)
        from core.event_bus import get_event_bus

        queue = await get_event_bus().subscribe(topic)
        history = self.retained(topic, profile=profile)
        return queue, history

    def retained(self, topic: str, *, profile: QosProfile = DEFAULT) -> list[Sample]:
        """The samples a late joiner is entitled to, lifespan applied."""
        if profile.durability is not Durability.TRANSIENT_LOCAL:
            return []
        with self._lock:
            state = self._topics.get(topic)
            if state is None or state.profile.durability is not Durability.TRANSIENT_LOCAL:
                return []
            samples = list(state.retained)
            lifespan = state.profile.lifespan_s
        now = time.time()
        fresh = [s for s in samples if not s.expired(lifespan, now)]
        if profile.history is History.KEEP_LAST:
            return fresh[-max(1, profile.depth) :]
        return fresh

    async def stream(
        self, topic: str, *, profile: QosProfile = DEFAULT
    ) -> AsyncIterator[Any]:
        """Retained history first, then live samples — the natural order."""
        queue, history = await self.subscribe(topic, profile=profile)
        for sample in history:
            yield sample.data
        stream_closed = asyncio.Event()
        # The wait is bounded by the subscriber's own declared deadline
        # where it has one, and by a poll interval otherwise. A stream
        # that blocks forever cannot notice that its deadline has already
        # been missed — which is precisely the absence this profile
        # exists to make visible — and cannot be shut down while quiet.
        budget = profile.deadline_s if profile.deadline_s > 0 else STREAM_POLL_S
        try:
            while not stream_closed.is_set():
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=budget)
                except TimeoutError:
                    if profile.deadline_s > 0:
                        self._fire_deadline(topic, budget)
                    continue
                yield event[1] if isinstance(event, tuple) else event
        finally:
            stream_closed.set()
            from core.event_bus import get_event_bus

            with contextlib.suppress(Exception):
                await get_event_bus().unsubscribe(topic, queue)

    # ── deadline / liveliness ─────────────────────────────────────────
    def on_deadline_missed(self, topic: str, callback: Callable[[str, float], None]) -> None:
        with self._lock:
            self._deadline_callbacks.setdefault(topic, []).append(callback)

    def on_liveliness_lost(self, topic: str, callback: Callable[[str], None]) -> None:
        with self._lock:
            self._liveliness_callbacks.setdefault(topic, []).append(callback)

    def _fire_deadline(self, topic: str, gap_s: float) -> None:
        with self._lock:
            callbacks = list(self._deadline_callbacks.get(topic, ()))
        logger.warning(
            "deadline missed on %r: %.2fs since the previous sample", topic, gap_s
        )
        for callback in callbacks:
            with contextlib.suppress(Exception):
                callback(topic, gap_s)

    def check_liveliness(self) -> list[str]:
        """Sweep for publishers that stopped asserting. Absence is the signal."""
        now = time.time()
        lost: list[str] = []
        with self._lock:
            states = list(self._topics.items())
        for topic, state in states:
            lease = state.profile.liveliness_lease_s
            if lease <= 0 or not state.last_assert_at:
                continue
            if state.alive and (now - state.last_assert_at) > lease:
                with self._lock:
                    state.alive = False
                    state.liveliness_losses += 1
                    callbacks = list(self._liveliness_callbacks.get(topic, ()))
                lost.append(topic)
                logger.warning(
                    "liveliness lost on %r: no assertion for %.1fs (lease %.1fs)",
                    topic,
                    now - state.last_assert_at,
                    lease,
                )
                for callback in callbacks:
                    with contextlib.suppress(Exception):
                        callback(topic)
        return lost

    def check_deadlines(self) -> list[str]:
        """Sweep for topics that have gone quiet past their deadline."""
        now = time.time()
        missed: list[str] = []
        with self._lock:
            states = list(self._topics.items())
        for topic, state in states:
            deadline = state.profile.deadline_s
            if deadline <= 0 or not state.last_published_at:
                continue
            gap = now - state.last_published_at
            if gap > deadline:
                with self._lock:
                    state.deadline_misses += 1
                missed.append(topic)
                self._fire_deadline(topic, gap)
        return missed

    # ── reporting ─────────────────────────────────────────────────────
    def _state(self, topic: str) -> TopicState:
        with self._lock:
            state = self._topics.get(topic)
            if state is None:
                state = TopicState(profile=DEFAULT, retained=deque(maxlen=DEFAULT.depth))
                self._topics[topic] = state
            return state

    def report(self) -> dict[str, Any]:
        with self._lock:
            topics = {
                topic: {
                    "profile": state.profile.to_dict(),
                    "published": state.published,
                    "dropped": state.dropped,
                    "retained": len(state.retained),
                    "deadline_misses": state.deadline_misses,
                    "liveliness_losses": state.liveliness_losses,
                    "alive": state.alive,
                    "last_published_age_s": (
                        round(time.time() - state.last_published_at, 2)
                        if state.last_published_at
                        else None
                    ),
                }
                for topic, state in sorted(self._topics.items())
            }
            mismatches = list(self._mismatches[-8:])
        return {
            "topics": topics,
            "topic_count": len(topics),
            "transient_local": [
                t
                for t, e in topics.items()
                if e["profile"]["durability"] == "transient_local"
            ],
            "not_alive": [t for t, e in topics.items() if not e["alive"]],
            "qos_mismatches": mismatches,
        }

    def reset_for_test(self) -> None:
        with self._lock:
            self._topics.clear()
            self._mismatches.clear()
            self._deadline_callbacks.clear()
            self._liveliness_callbacks.clear()


_BUS = QosBus()


def get_qos_bus() -> QosBus:
    return _BUS


def declare_topic(topic: str, profile: QosProfile = DEFAULT) -> TopicState:
    return _BUS.declare_publisher(topic, profile)


async def publish(topic: str, data: Any, *, profile: QosProfile | None = None) -> bool:
    return await _BUS.publish(topic, data, profile=profile)


async def subscribe(
    topic: str, *, profile: QosProfile = DEFAULT
) -> tuple[asyncio.Queue, list[Sample]]:
    return await _BUS.subscribe(topic, profile=profile)


def qos_report() -> dict[str, Any]:
    return _BUS.report()


def reset_qos_for_test() -> None:
    _BUS.reset_for_test()


def with_depth(profile: QosProfile, depth: int) -> QosProfile:
    return replace(profile, depth=max(1, depth))


__all__ = [
    "COMMAND",
    "DEFAULT",
    "HEARTBEAT",
    "SENSOR_DATA",
    "STATE",
    "Durability",
    "History",
    "QosBus",
    "QosProfile",
    "Reliability",
    "Sample",
    "TopicState",
    "declare_topic",
    "get_qos_bus",
    "publish",
    "qos_report",
    "reset_qos_for_test",
    "subscribe",
    "with_depth",
]
