"""Aura input bus: typed pub/sub over a bounded priority queue.

Three contracts are load-bearing and all three were open before CP126:

* **Admission precedes effect.** Subscribers used to run before the queue was
  written, so a full queue dropped "only the queued copy" — after every side
  effect had already happened.
* **The bus owns ordering.** The queue key was the caller's own timestamp, so
  any publisher could position itself in the queue. It is now a bus-assigned
  monotonic sequence.
* **A receipt must be true.** "Moved to DLQ after 3 failures" counted *three
  failing subscribers in one delivery*, never a retry, and could append the
  same mutable event repeatedly.

CP126 03d556a3 / 1ae03583 / 086d721e / c280c642 / 2ea4046f / f93e3dfd.
"""
from __future__ import annotations

import itertools
import json
import logging
import math
import queue
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from enum import Enum, IntEnum, auto
from typing import Any

from core.runtime.errors import record_degradation

logger = logging.getLogger("Core.Events")

#: Envelope version. Present on every event so a consumer can tell what it is
#: reading (CP126 03d556a3).
EVENT_SCHEMA_VERSION = "aura.bus.event.v2"

MAX_PAYLOAD_BYTES = 256 * 1024
MAX_TOPIC_CHARS = 256
MAX_SOURCE_CHARS = 128
#: An event timestamped more than this far from now is not usable as a clock
#: reading; it is replaced and the original is kept in the payload.
MAX_TIMESTAMP_SKEW_S = 365 * 24 * 3600.0
#: Bound on the idempotency record. Memory-only: see InputBus.next().
MAX_SEEN_KEYS = 8192
MAX_DLQ = 500
#: Delivery attempts before an event with a failing subscriber is dead-lettered.
MAX_DELIVERY_ATTEMPTS = 3
#: Wall-clock budget for one subscriber callback on the async path.
DEFAULT_CALLBACK_TIMEOUT_S = 5.0


class EventPriority(IntEnum):
    """High value = high priority"""

    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


class EventType(Enum):
    USER_MESSAGE = auto()
    MIC_TRANSCRIPT = auto()
    TRAINING_STATUS = auto()
    SYSTEM = auto()
    DOWNLOAD_PROGRESS = auto()
    ERROR = auto()
    HEALTH_ALERT = auto()
    SELF_MOD_PROPOSAL = auto()
    SKILL_EXECUTED = auto()
    STATE_CHANGE = auto()


@dataclass(frozen=True, order=True)
class Event:
    """An immutable event envelope.

    CP126 086d721e: the mutable instance was handed to every subscriber in turn
    and then enqueued, so a callback could rewrite the type, payload, priority,
    timestamp or source that later subscribers and the queue consumer saw —
    after normalization had already run.
    """

    priority: EventPriority = field(default=EventPriority.NORMAL, compare=True)
    ts: float = field(default_factory=lambda: time.time(), compare=True)
    type: EventType = field(default=EventType.SYSTEM, compare=False)
    topic: str = field(default="", compare=False)
    payload: dict[str, Any] = field(default_factory=dict, compare=False)
    source: str = field(default="", compare=False)
    #: Delivery attempts made for this event, set by the bus — never by a
    #: subscriber failure (CP126 c280c642).
    retry_count: int = field(default=0, compare=False)
    event_id: str = field(default="", compare=False)
    schema_version: str = field(default=EVENT_SCHEMA_VERSION, compare=False)
    idempotency_key: str = field(default="", compare=False)
    #: Anything normalization had to repair, kept with the event.
    envelope_faults: tuple[str, ...] = field(default=(), compare=False)

    def __post_init__(self) -> None:
        if not self.event_id:
            object.__setattr__(self, "event_id", uuid.uuid4().hex)

    @property
    def dedupe_key(self) -> str:
        return self.idempotency_key or self.event_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "schema_version": self.schema_version,
            "type": self.type.name,
            "topic": self.topic,
            "priority": int(self.priority),
            "ts": self.ts,
            "source": self.source,
            "payload": self.payload,
            "retry_count": self.retry_count,
            "idempotency_key": self.idempotency_key,
            "envelope_faults": list(self.envelope_faults),
        }


@dataclass(frozen=True)
class DeliveryReceipt:
    """What publishing actually achieved (CP126 1ae03583)."""

    event_id: str
    queued: bool
    notified: int = 0
    failed: tuple[str, ...] = ()
    dropped_reason: str = ""
    duplicate: bool = False
    dead_lettered: bool = False

    @property
    def ok(self) -> bool:
        return self.queued and not self.failed

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "queued": self.queued,
            "notified": self.notified,
            "failed": list(self.failed),
            "dropped_reason": self.dropped_reason,
            "duplicate": self.duplicate,
            "dead_lettered": self.dead_lettered,
            "ok": self.ok,
        }


def _coerce_priority(value: Any) -> EventPriority:
    """Normalize external priority values before they reach the queue."""
    if isinstance(value, EventPriority):
        return value
    if isinstance(value, str):
        key = value.strip().upper()
        if key in EventPriority.__members__:
            return EventPriority[key]
    try:
        return EventPriority(int(value))
    except (TypeError, ValueError):
        logger.debug("Invalid event priority %r; defaulting to NORMAL", value)
        return EventPriority.NORMAL


def _coerce_type(value: Any) -> tuple[EventType, str]:
    """An EventType, plus a fault when the caller asked for something else.

    CP126 03d556a3: an unknown type was silently demoted to SYSTEM, so a typo'd
    or hostile type name became an ordinary system event with no trace.
    """
    if isinstance(value, EventType):
        return value, ""
    if isinstance(value, str):
        key = value.strip().upper()
        if key in EventType.__members__:
            return EventType[key], ""
        return EventType.SYSTEM, f"unknown event type {value!r} demoted to SYSTEM"
    if value is None:
        return EventType.SYSTEM, ""
    return EventType.SYSTEM, f"event type {type(value).__name__} demoted to SYSTEM"


def _bounded_text(value: Any, limit: int, name: str) -> tuple[str, str]:
    text = "" if value is None else str(value)
    if len(text) > limit:
        return text[:limit], f"{name} truncated from {len(text)} chars"
    return text, ""


def _finite_timestamp(value: Any) -> tuple[float, str]:
    now = time.time()
    try:
        ts = float(value)
    except (TypeError, ValueError):
        return now, f"non-numeric timestamp {value!r} replaced"
    if math.isnan(ts) or math.isinf(ts):
        return now, f"non-finite timestamp {ts} replaced"
    if abs(ts - now) > MAX_TIMESTAMP_SKEW_S:
        return now, f"timestamp {ts} is implausible; replaced"
    return ts, ""


def _bounded_payload(payload: Any) -> tuple[dict[str, Any], str]:
    if not isinstance(payload, dict):
        payload = {"data": payload}
    try:
        size = len(json.dumps(payload, default=str).encode("utf-8"))
    except (TypeError, ValueError, RecursionError):
        size = len(str(payload))
    if size <= MAX_PAYLOAD_BYTES:
        return payload, ""
    preview = str(payload)[:512]
    return (
        {"_truncated": True, "_original_bytes": size, "_preview": preview},
        f"payload of {size} bytes exceeds the {MAX_PAYLOAD_BYTES}-byte bound",
    )


class InputBus:
    """Thread-safe event multiplexer with pub/sub + priority queue.

    Includes a Dead Letter Queue (DLQ) for failed event processing.

    Durability note: the queue, the DLQ and the idempotency record are all in
    memory. A process loss discards in-flight delivery state; this bus provides
    at-most-once delivery with acknowledgement, not durable messaging.
    """

    def __init__(
        self,
        maxsize: int = 2000,
        *,
        retry_failed_subscribers: bool = False,
        callback_timeout_s: float = DEFAULT_CALLBACK_TIMEOUT_S,
    ):
        # PriorityQueue pops the lowest value first, so the key is
        # (-priority, sequence): CRITICAL sorts before LOW, and the bus's own
        # monotonic sequence breaks ties FIFO. CP126 03d556a3: the tiebreaker
        # used to be the caller's timestamp, which let any publisher choose its
        # position in the queue.
        self._q: queue.PriorityQueue = queue.PriorityQueue(maxsize=maxsize)
        self._sequence = itertools.count()
        self._dlq: list[dict[str, Any]] = []
        self._dlq_ids: set[str] = set()
        self._subscribers: dict[EventType, list[Callable[[Event], None]]] = {}
        self._lock = threading.Lock()
        self._dlq_lock = threading.Lock()
        self._seen: OrderedDict[str, float] = OrderedDict()
        self._in_flight: dict[str, Event] = {}
        self._failures: dict[str, dict[str, int]] = {}
        self._closed = False
        self.retry_failed_subscribers = bool(retry_failed_subscribers)
        self.callback_timeout_s = float(callback_timeout_s)
        # A single notifier thread keeps async delivery in submission order;
        # CP126 1ae03583 called out callbacks reordering across events.
        self._notifier = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="Aura.Events.Notify"
        )
        # Separate pool so one slow callback cannot occupy the ordered notifier.
        self._executor = ThreadPoolExecutor(
            max_workers=8, thread_name_prefix="Aura.Events"
        )

    # -- publication ------------------------------------------------------
    def publish(
        self,
        event: Event | dict[str, Any] | str,
        block: bool = False,
        timeout: float | None = None,
    ) -> DeliveryReceipt:
        """Admit an event, then notify subscribers synchronously.

        CP126 1ae03583: notification used to run first, so a rejected event
        still fired every side effect. Nothing observes an event the queue
        refused.
        """
        event_obj = self._normalize_event(event)
        receipt = self._enqueue(event_obj, block, timeout)
        if not receipt.queued:
            return receipt
        return self._notify_subscribers(event_obj, receipt)

    def publish_async(
        self,
        event: Event | dict[str, Any] | str,
        block: bool = False,
        timeout: float | None = None,
    ) -> DeliveryReceipt:
        """Admit an event, then notify subscribers on the ordered notifier."""
        event_obj = self._normalize_event(event)
        receipt = self._enqueue(event_obj, block, timeout)
        if not receipt.queued:
            return receipt
        if self._closed:
            return replace(receipt, dropped_reason="bus_closed")
        try:
            self._notifier.submit(self._notify_subscribers, event_obj, receipt)
        except RuntimeError as exc:  # executor already shut down
            record_degradation("events", exc)
            return replace(receipt, dropped_reason="notifier_unavailable")
        return receipt

    def emit(
        self,
        event_type: EventType,
        payload: dict[str, Any],
        source: str = "",
        priority: EventPriority = EventPriority.NORMAL,
        *,
        idempotency_key: str = "",
    ) -> DeliveryReceipt:
        """Convenience: create and publish an Event in one call."""
        event = Event(
            type=event_type,
            payload=payload,
            source=source,
            priority=priority,
            idempotency_key=idempotency_key,
        )
        return self.publish(event)

    # -- normalization ----------------------------------------------------
    def _normalize_event(
        self, event_input: Event | dict[str, Any] | str | tuple
    ) -> Event:
        """Every input becomes a validated, bus-stamped envelope.

        CP126 03d556a3: an existing Event bypassed normalization entirely, so a
        caller-built envelope reached the queue with an arbitrary timestamp,
        unbounded payload and no identity.
        """
        if isinstance(event_input, Event):
            return self._validated(event_input)

        if isinstance(event_input, tuple) and len(event_input) == 2:
            topic, payload = event_input
            return self._validated(
                Event(
                    type=EventType.SYSTEM,
                    topic=str(topic),
                    payload=payload if isinstance(payload, dict) else {"data": payload},
                    source="tuple_input",
                )
            )

        if isinstance(event_input, dict):
            event_type, type_fault = _coerce_type(event_input.get("type", "SYSTEM"))
            # CP126 03d556a3: the whole envelope used to become the payload,
            # so routing metadata was indistinguishable from event data.
            raw_payload = event_input.get("payload")
            if not isinstance(raw_payload, dict):
                raw_payload = {
                    key: value
                    for key, value in event_input.items()
                    if key not in {"type", "topic", "priority", "source", "idempotency_key", "ts"}
                }
            candidate = Event(
                type=event_type,
                topic=str(event_input.get("topic", "") or ""),
                payload=raw_payload,
                source=str(event_input.get("source", "normalized_dict") or "normalized_dict"),
                priority=_coerce_priority(event_input.get("priority", EventPriority.NORMAL)),
                idempotency_key=str(event_input.get("idempotency_key", "") or ""),
            )
            return self._validated(candidate, extra_faults=(type_fault,) if type_fault else ())

        if isinstance(event_input, str):
            return self._validated(
                Event(
                    type=EventType.SYSTEM,
                    topic=event_input if "/" in event_input else "",
                    payload={"message": event_input},
                    source="normalized_string",
                )
            )

        return self._validated(
            Event(payload={"raw_data": str(event_input)}, source="unknown_input")
        )

    @staticmethod
    def _validated(event: Event, *, extra_faults: tuple[str, ...] = ()) -> Event:
        faults: list[str] = list(event.envelope_faults) + [f for f in extra_faults if f]
        priority = _coerce_priority(event.priority)
        event_type, type_fault = _coerce_type(event.type)
        if type_fault:
            faults.append(type_fault)
        ts, ts_fault = _finite_timestamp(event.ts)
        if ts_fault:
            faults.append(ts_fault)
        topic, topic_fault = _bounded_text(event.topic, MAX_TOPIC_CHARS, "topic")
        if topic_fault:
            faults.append(topic_fault)
        source, source_fault = _bounded_text(event.source, MAX_SOURCE_CHARS, "source")
        if source_fault:
            faults.append(source_fault)
        payload, payload_fault = _bounded_payload(event.payload)
        if payload_fault:
            faults.append(payload_fault)
        try:
            retry_count = max(0, int(event.retry_count))
        except (TypeError, ValueError):
            retry_count = 0
            faults.append("retry_count was not an integer")

        return replace(
            event,
            priority=priority,
            type=event_type,
            ts=ts,
            topic=topic,
            source=source,
            payload=payload,
            retry_count=retry_count,
            schema_version=EVENT_SCHEMA_VERSION,
            envelope_faults=tuple(faults),
        )

    # -- admission --------------------------------------------------------
    def _enqueue(
        self, event: Event, block: bool = False, timeout: float | None = None
    ) -> DeliveryReceipt:
        if self._closed:
            return DeliveryReceipt(event.event_id, queued=False, dropped_reason="bus_closed")
        if self._is_duplicate(event):
            logger.debug("Duplicate event %s suppressed", event.dedupe_key)
            return DeliveryReceipt(
                event.event_id, queued=False, dropped_reason="duplicate", duplicate=True
            )
        try:
            self._q.put(
                (-int(event.priority), next(self._sequence), event),
                block=block,
                timeout=timeout,
            )
        except queue.Full:
            logger.warning("Event queue full — dropping event type=%s", event.type.name)
            return DeliveryReceipt(
                event.event_id, queued=False, dropped_reason="queue_full"
            )
        return DeliveryReceipt(event.event_id, queued=True)

    def _is_duplicate(self, event: Event) -> bool:
        if not event.idempotency_key:
            return False
        with self._lock:
            if event.idempotency_key in self._seen:
                return True
            self._seen[event.idempotency_key] = time.time()
            while len(self._seen) > MAX_SEEN_KEYS:
                self._seen.popitem(last=False)
        return False

    # -- consumption (CP126 f93e3dfd) -------------------------------------
    def next(self, timeout: float | None = 0.1) -> Event | None:
        """Poll the next highest-priority event and hold it in flight.

        The event stays in ``in_flight`` until ``ack()`` or ``nack()``. This is
        an in-memory acknowledgement contract: it survives consumer *failure*,
        not process loss.
        """
        try:
            _, _, event = self._q.get(timeout=timeout)
        except queue.Empty as exc:
            logger.debug(
                "no event arrived inside the poll window (%s: %s)",
                type(exc).__name__,
                exc,
            )
            return None
        with self._lock:
            self._in_flight[event.event_id] = event
        return event

    def ack(self, event: Event | str) -> bool:
        """Confirm an event was handled; release it from the in-flight set."""
        event_id = event.event_id if isinstance(event, Event) else str(event)
        with self._lock:
            held = self._in_flight.pop(event_id, None)
            self._failures.pop(event_id, None)
        if held is None:
            return False
        self._task_done()
        return True

    def nack(
        self, event: Event | str, *, requeue: bool = True, reason: str = ""
    ) -> DeliveryReceipt:
        """Report that an event could not be handled.

        Requeues it with an incremented delivery attempt until the attempt
        budget is spent, then dead-letters it — which is what "after N
        failures" is supposed to mean.
        """
        event_id = event.event_id if isinstance(event, Event) else str(event)
        with self._lock:
            held = self._in_flight.pop(event_id, None)
        if held is None:
            held = event if isinstance(event, Event) else None
        if held is None:
            return DeliveryReceipt(event_id, queued=False, dropped_reason="unknown_event")
        self._task_done()

        attempts = held.retry_count + 1
        if not requeue or attempts >= MAX_DELIVERY_ATTEMPTS:
            detail = f"nacked after {attempts} delivery attempt(s)"
            self._dead_letter(
                held,
                f"{reason}; {detail}" if reason else detail,
                failed_subscribers=(),
                attempts=attempts,
            )
            return DeliveryReceipt(
                event_id, queued=False, dropped_reason="dead_lettered", dead_lettered=True
            )
        retried = replace(held, retry_count=attempts)
        return self._enqueue(retried)

    def _task_done(self) -> None:
        try:
            self._q.task_done()
        except ValueError:
            # More task_done() calls than items; the caller acked twice.
            logger.debug("task_done called without a matching get")

    @property
    def in_flight_count(self) -> int:
        with self._lock:
            return len(self._in_flight)

    # -- subscription -----------------------------------------------------
    def subscribe(self, event_type: EventType, callback: Callable[[Event], None]) -> None:
        """Register a callback for a specific event type."""
        with self._lock:
            self._subscribers.setdefault(event_type, []).append(callback)
            logger.debug("Subscriber registered for %s", event_type.name)

    def unsubscribe(self, event_type: EventType, callback: Callable[[Event], None]) -> bool:
        with self._lock:
            callbacks = self._subscribers.get(event_type, [])
            if callback in callbacks:
                callbacks.remove(callback)
                return True
        return False

    def _notify_subscribers(
        self, event: Event, receipt: DeliveryReceipt | None = None
    ) -> DeliveryReceipt:
        """Call every subscriber for this event type in isolation."""
        receipt = receipt or DeliveryReceipt(event.event_id, queued=True)
        with self._lock:
            callbacks = list(self._subscribers.get(event.type, []))

        failed: list[str] = []
        notified = 0
        for callback in callbacks:
            name = getattr(callback, "__name__", repr(callback))
            # CP126 086d721e: each subscriber gets its own shallow copy, so a
            # callback that edits the payload cannot change what the next
            # subscriber or the queue consumer sees.
            if self._invoke(callback, replace(event, payload=dict(event.payload)), name):
                notified += 1
            else:
                failed.append(name)

        dead_lettered = False
        if failed:
            dead_lettered = self._handle_failure(event, tuple(failed))
        return replace(
            receipt,
            notified=notified,
            failed=tuple(failed),
            dead_lettered=dead_lettered,
        )

    def _invoke(self, callback: Callable[[Event], None], event: Event, name: str) -> bool:
        """Run one callback, isolating every failure it can produce.

        CP126 2ea4046f: only four exception classes were caught, so any other
        subscriber exception aborted the rest of a synchronous publish or sat
        unobserved in an executor future.
        """
        started = time.monotonic()
        try:
            callback(event)
        except Exception as exc:  # noqa: BLE001 - a subscriber must not take the bus down
            record_degradation("events", exc)
            logger.error(
                "Subscriber %s failed on %s: %s", name, event.type.name, exc, exc_info=False
            )
            return False
        finally:
            elapsed = time.monotonic() - started
            if elapsed > self.callback_timeout_s:
                # A running thread cannot be preempted; the overrun is reported
                # rather than silently absorbed.
                logger.warning(
                    "Subscriber %s took %.2fs on %s (budget %.2fs)",
                    name, elapsed, event.type.name, self.callback_timeout_s,
                )
        return True

    # -- failure handling (CP126 c280c642) --------------------------------
    def _handle_failure(self, event: Event, failed: tuple[str, ...]) -> bool:
        """Record failing subscribers and dead-letter the event once.

        The old implementation incremented ``retry_count`` once per failing
        subscriber, so three failing subscribers in a single delivery produced
        a "moved to DLQ after 3 failures" receipt with no retry ever attempted,
        and later failures appended the same mutable event again.
        """
        with self._lock:
            counts = self._failures.setdefault(event.event_id, {})
            for name in failed:
                counts[name] = counts.get(name, 0) + 1
            attempts = event.retry_count + 1

        if self.retry_failed_subscribers and attempts < MAX_DELIVERY_ATTEMPTS:
            self._schedule_retry(event, failed, attempts)
            return False

        reason = (
            f"{len(failed)} subscriber(s) failed on delivery attempt {attempts}"
            + ("" if self.retry_failed_subscribers else "; subscriber retries are disabled")
        )
        return self._dead_letter(event, reason, failed_subscribers=failed, attempts=attempts)

    def _schedule_retry(
        self, event: Event, failed: tuple[str, ...], attempts: int
    ) -> None:
        """Re-run only the failing callbacks, once, after a bounded backoff."""
        delay = min(2.0, 0.05 * (2 ** attempts))

        def _retry() -> None:
            time.sleep(delay)
            with self._lock:
                callbacks = [
                    cb
                    for cb in self._subscribers.get(event.type, [])
                    if getattr(cb, "__name__", repr(cb)) in failed
                ]
            retried = replace(event, retry_count=attempts)
            still_failing = [
                getattr(cb, "__name__", repr(cb))
                for cb in callbacks
                if not self._invoke(
                    cb, replace(retried, payload=dict(retried.payload)),
                    getattr(cb, "__name__", repr(cb)),
                )
            ]
            if still_failing:
                self._handle_failure(retried, tuple(still_failing))

        try:
            self._executor.submit(_retry)
        except RuntimeError as exc:
            record_degradation("events", exc)
            self._dead_letter(
                event, f"retry could not be scheduled: {exc}", failed, attempts
            )

    def _dead_letter(
        self,
        event: Event,
        reason: str,
        failed_subscribers: tuple[str, ...],
        attempts: int,
    ) -> bool:
        with self._dlq_lock:
            if event.event_id in self._dlq_ids:
                # CP126 c280c642: the same event could be appended repeatedly.
                return False
            if len(self._dlq) >= MAX_DLQ:
                logger.error("DLQ full. Dropping failed event type=%s", event.type.name)
                return False
            self._dlq.append(
                {
                    "event": event,
                    "reason": reason,
                    "failed_subscribers": list(failed_subscribers),
                    "delivery_attempts": attempts,
                    "dead_lettered_at": time.time(),
                }
            )
            self._dlq_ids.add(event.event_id)
        logger.critical(
            "Event type=%s moved to the Dead Letter Queue (%s).", event.type.name, reason
        )
        return True

    def replay_dlq(self, *, limit: int = 0) -> list[DeliveryReceipt]:
        """Re-admit dead-lettered events, newest last. Returns their receipts."""
        with self._dlq_lock:
            entries = list(self._dlq if limit <= 0 else self._dlq[:limit])
            for entry in entries:
                self._dlq.remove(entry)
                self._dlq_ids.discard(entry["event"].event_id)
        receipts = []
        for entry in entries:
            event = replace(entry["event"], retry_count=0)
            receipts.append(self._enqueue(event))
        return receipts

    # -- introspection ----------------------------------------------------
    @property
    def subscriber_count(self) -> int:
        """Total number of registered subscribers across all event types."""
        with self._lock:
            return sum(len(cbs) for cbs in self._subscribers.values())

    @property
    def queue_depth(self) -> int:
        return self._q.qsize()

    def get_dlq_stats(self) -> dict[str, Any]:
        """Report on dead events."""
        with self._dlq_lock:
            return {
                "count": len(self._dlq),
                "types": [entry["event"].type.name for entry in self._dlq[-10:]],
                "reasons": [entry["reason"] for entry in self._dlq[-10:]],
            }

    def dlq_events(self) -> list[dict[str, Any]]:
        with self._dlq_lock:
            return [dict(entry) for entry in self._dlq]

    # -- lifecycle (CP126 2ea4046f) ---------------------------------------
    def shutdown(self, timeout: float = 5.0) -> bool:
        """Stop accepting work and release the executors within a deadline.

        ``shutdown(wait=True)`` had no deadline, so process teardown could hang
        behind a blocking callback. Pending work is cancelled and the join is
        bounded; the return value says whether everything drained.
        """
        self._closed = True
        drained = True
        for executor in (self._notifier, self._executor):
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except (RuntimeError, TypeError) as exc:
                record_degradation("events", exc)
                drained = False
        deadline = time.monotonic() + max(0.0, float(timeout))
        for executor in (self._notifier, self._executor):
            for thread in list(getattr(executor, "_threads", ()) or ()):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    drained = False
                    break
                thread.join(timeout=remaining)
                if thread.is_alive():
                    logger.warning(
                        "Event executor thread %s did not stop within the shutdown budget",
                        thread.name,
                    )
                    drained = False
        return drained
