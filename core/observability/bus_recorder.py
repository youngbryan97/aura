"""core/observability/bus_recorder.py — record and replay the event bus.

Clean-room adoption of `rosbag2`: capture every message on the bus with
its topic and timestamp, then replay it.

Debugging a cognitive runtime has a specific difficulty. The interesting
failures are not crashes with a stack trace; they are *decisions* that
came out wrong, produced by a specific sequence of events that will never
occur in that order again. By the time you know a turn went wrong, the
inputs that produced it are gone. Reasoning about it means reading logs
that record what each component decided to say about itself, which is not
the same as what it saw.

A bag records what it *saw*. With one, a bad turn becomes reproducible:
replay the bag and the same inputs arrive in the same order, which turns
"it did something strange last Tuesday" into a test case. Combined with
the pass manager's `-opt-bisect-limit`, a recorded bad turn can be
bisected down to the phase that ruined it.

Two modes, and the always-on one is the point:

* **The ring** is always recording, in memory, bounded. Nothing has to
  predict that something interesting is about to happen — the last N
  seconds are always there, and a degradation, a crash, or a lockdep
  splat dumps them. This is the same bet the flight recorder makes about
  mind-moments, applied to the bus.
* **A bag file** is an explicit, unbounded-in-time recording of chosen
  topics, for when you already know what you are hunting.

Recording is off the hot path by construction: the ring costs a deque
append, and file writes go through the async lane.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import threading
import time
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("Aura.BusRecorder")

#: Default ring capacity. ~8k messages covers minutes of normal traffic
#: and costs a few MB.
DEFAULT_RING_CAPACITY = 8192


@dataclass(frozen=True)
class BagMessage:
    topic: str
    at: float
    monotonic: float
    payload: Any
    sequence: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "topic": self.topic,
            "at": self.at,
            "monotonic": self.monotonic,
            "sequence": self.sequence,
            "payload": _summarize(self.payload),
        }


def _summarize(payload: Any, *, limit: int = 4096) -> Any:
    """Make a payload storable without dragging in live object graphs."""
    try:
        encoded = json.dumps(payload, default=repr)
    except (TypeError, ValueError):
        return {"repr": repr(payload)[:limit]}
    if len(encoded) <= limit:
        return json.loads(encoded)
    return {"truncated": True, "bytes": len(encoded), "head": encoded[:limit]}


class BusRecorder:
    """The always-on ring plus optional file recording."""

    def __init__(self, *, capacity: int = DEFAULT_RING_CAPACITY) -> None:
        self._lock = threading.Lock()
        self._ring: deque[BagMessage] = deque(maxlen=capacity)
        self._sequence = 0
        self._topics: set[str] = set()
        self._exclude: set[str] = set()
        self._recording = True
        self._file_topics: set[str] | None = None
        self._file_path: Path | None = None
        self._file_buffer: list[str] = []
        self.captured = 0
        self.dropped = 0
        self.dumps = 0

    # ── capture ───────────────────────────────────────────────────────
    def record(self, topic: str, payload: Any) -> None:
        """Called on every publish. Must stay cheap."""
        if not self._recording or topic in self._exclude:
            return
        now = time.time()
        with self._lock:
            self._sequence += 1
            message = BagMessage(
                topic=topic,
                at=now,
                monotonic=time.monotonic(),
                payload=payload,
                sequence=self._sequence,
            )
            if len(self._ring) == self._ring.maxlen:
                self.dropped += 1
            self._ring.append(message)
            self._topics.add(topic)
            self.captured += 1
            if self._file_path is not None and (
                self._file_topics is None or topic in self._file_topics
            ):
                self._file_buffer.append(
                    json.dumps(message.to_dict(), separators=(",", ":"))
                )

    def exclude(self, *topics: str) -> None:
        """Keep high-rate noise out of the ring so signal survives in it."""
        with self._lock:
            self._exclude.update(topics)

    def pause(self) -> None:
        self._recording = False

    def resume(self) -> None:
        self._recording = True

    # ── the ring ──────────────────────────────────────────────────────
    def window(self, seconds: float = 60.0, *, topics: tuple[str, ...] = ()) -> list[BagMessage]:
        cutoff = time.monotonic() - seconds
        wanted = set(topics)
        with self._lock:
            messages = list(self._ring)
        return [
            m
            for m in messages
            if m.monotonic >= cutoff and (not wanted or m.topic in wanted)
        ]

    async def dump(
        self,
        *,
        reason: str,
        seconds: float = 60.0,
        topics: tuple[str, ...] = (),
        directory: Path | None = None,
    ) -> Path | None:
        """Write the recent ring to disk. The whole point of always-on.

        Called by degradation handlers, crash forensics, and lockdep
        splats — anything that discovers, after the fact, that the last
        minute was interesting.
        """
        messages = self.window(seconds, topics=topics)
        if not messages:
            return None
        target = _bag_dir(directory) / f"bus_{int(time.time())}_{_slug(reason)}.jsonl"
        header = {
            "kind": "aura_bag",
            "version": 1,
            "reason": reason,
            "window_s": seconds,
            "messages": len(messages),
            "topics": sorted({m.topic for m in messages}),
            "written_at": time.time(),
        }
        body = "\n".join(
            [json.dumps(header, separators=(",", ":"))]
            + [json.dumps(m.to_dict(), separators=(",", ":")) for m in messages]
        )
        try:
            from core.governance_context import local_internal_governed_scope
            from core.runtime.file_write_gateway import get_file_write_gateway

            with local_internal_governed_scope("bus_recorder.dump"):
                gateway = get_file_write_gateway()
                await gateway.ensure_directory_async(
                    target.parent,
                    source="bus_recorder.dump",
                )
                await gateway.write_text_async(
                    target,
                    body + "\n",
                    durable=False,
                    source="bus_recorder.dump",
                )
        except Exception:  # noqa: BLE001 — evidence, never a dependency
            logger.warning("bus bag dump failed", exc_info=True)
            return None
        with self._lock:
            self.dumps += 1
        logger.info("📼 bus bag written: %s (%d messages, %s)", target, len(messages), reason)
        return target

    # ── file recording ────────────────────────────────────────────────
    def start_file(self, path: Path, *, topics: tuple[str, ...] = ()) -> None:
        with self._lock:
            self._file_path = path
            self._file_topics = set(topics) or None
            self._file_buffer.clear()

    async def stop_file(self) -> Path | None:
        with self._lock:
            path = self._file_path
            buffer, self._file_buffer = self._file_buffer, []
            self._file_path = None
            self._file_topics = None
        if path is None or not buffer:
            return None
        try:
            from core.governance_context import local_internal_governed_scope
            from core.runtime.file_write_gateway import get_file_write_gateway

            with local_internal_governed_scope("bus_recorder.file"):
                gateway = get_file_write_gateway()
                await gateway.ensure_directory_async(
                    path.parent,
                    source="bus_recorder.file",
                )
                await gateway.write_text_async(
                    path,
                    "\n".join(buffer) + "\n",
                    durable=True,
                    source="bus_recorder.file",
                )
        except Exception:  # noqa: BLE001
            logger.warning("bus bag file write failed", exc_info=True)
            return None
        return path

    def what_has_happened(self, *, most: int = 400) -> list[tuple[str, float]]:
        """The ring as a stream of named moments, newest last.

        The ring already holds what happened and when; nothing could read it
        as a SEQUENCE. `core/knowledge/temporal.py` induces ordering and
        recurrence from exactly this shape and had no live stream to read, so
        a regularity she has lived through a hundred times could not become a
        rule she can reason with.
        """

        with self._lock:
            ring = list(self._ring)
        return [(str(one.topic), float(one.at)) for one in ring[-max(1, int(most)) :]]

    def report(self) -> dict[str, Any]:
        with self._lock:
            ring = list(self._ring)
            topics = sorted(self._topics)
            excluded = sorted(self._exclude)
        span = (ring[-1].monotonic - ring[0].monotonic) if len(ring) > 1 else 0.0
        by_topic: dict[str, int] = {}
        for message in ring:
            by_topic[message.topic] = by_topic.get(message.topic, 0) + 1
        return {
            "recording": self._recording,
            "ring_size": len(ring),
            "ring_capacity": self._ring.maxlen,
            "ring_span_s": round(span, 2),
            "captured": self.captured,
            "dropped_from_ring": self.dropped,
            "dumps": self.dumps,
            "topics": topics,
            "excluded": excluded,
            "busiest": sorted(by_topic.items(), key=lambda kv: -kv[1])[:8],
            "file_recording": str(self._file_path) if self._file_path else None,
        }

    def reset_for_test(self) -> None:
        with self._lock:
            self._ring.clear()
            self._topics.clear()
            self._exclude.clear()
            self._file_buffer.clear()
            self._file_path = None
            self._file_topics = None
            self._sequence = 0
            self._recording = True
            self.captured = 0
            self.dropped = 0
            self.dumps = 0


def _bag_dir(directory: Path | None = None) -> Path:
    if directory is not None:
        return directory
    from core.config import config

    return Path(config.paths.data_dir) / "error_logs" / "bags"


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text)[:48].strip("_") or "dump"


class BagReader:
    """Replay a recorded bag."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.header: dict[str, Any] = {}
        self._messages: list[dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        with self.path.open(encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except ValueError:
                    logger.warning("bag %s line %d is not JSON; skipped", self.path, index)
                    continue
                if index == 0 and entry.get("kind") == "aura_bag":
                    self.header = entry
                    continue
                self._messages.append(entry)

    def __len__(self) -> int:
        return len(self._messages)

    def topics(self) -> list[str]:
        return sorted({str(m.get("topic", "")) for m in self._messages})

    def messages(self, *, topics: tuple[str, ...] = ()) -> Iterator[dict[str, Any]]:
        wanted = set(topics)
        for message in self._messages:
            if not wanted or message.get("topic") in wanted:
                yield message

    async def replay(
        self,
        *,
        rate: float = 0.0,
        topics: tuple[str, ...] = (),
        publish: Any = None,
    ) -> int:
        """Republish the bag onto the bus.

        ``rate`` 0 replays as fast as possible (the usual choice for
        reproducing a decision); 1.0 replays at the original wall-clock
        spacing (for anything timing-sensitive).
        """
        if publish is None:
            from core.event_bus import get_event_bus

            bus = get_event_bus()

            async def publish(topic: str, payload: Any) -> None:  # noqa: F811
                await bus.publish(topic, payload)

        count = 0
        previous: float | None = None
        for message in self.messages(topics=topics):
            monotonic = float(message.get("monotonic", 0.0) or 0.0)
            if rate > 0 and previous is not None:
                delay = max(0.0, (monotonic - previous) / rate)
                if delay:
                    await asyncio.sleep(min(delay, 5.0))
            previous = monotonic
            with contextlib.suppress(Exception):
                await publish(str(message.get("topic", "")), message.get("payload"))
                count += 1
        return count


_RECORDER = BusRecorder()


def get_bus_recorder() -> BusRecorder:
    return _RECORDER


def record(topic: str, payload: Any) -> None:
    _RECORDER.record(topic, payload)


def bus_recorder_report() -> dict[str, Any]:
    return _RECORDER.report()


def reset_bus_recorder_for_test() -> None:
    _RECORDER.reset_for_test()


__all__ = [
    "DEFAULT_RING_CAPACITY",
    "BagMessage",
    "BagReader",
    "BusRecorder",
    "bus_recorder_report",
    "get_bus_recorder",
    "record",
    "reset_bus_recorder_for_test",
]
