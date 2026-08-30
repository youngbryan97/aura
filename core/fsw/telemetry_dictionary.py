"""core/fsw/telemetry_dictionary.py — channels, limits, and the event log.

Clean-room adoption of F Prime's telemetry channels and EVRs (event
reports), plus OpenMCT's limit-evaluation model.

A spacecraft cannot be debugged interactively, so its ground system is
built on a *dictionary*: every value the vehicle can report has an ID, a
type, a unit, and declared limits; every notable occurrence is an event
with an ID, a severity, and a format string. The dictionary is generated
from the flight software, so the ground display and the vehicle can never
disagree about what channel 0x1042 means.

Aura is in the same position more often than it looks. The live runtime
cannot be stepped through; a stall or a bad turn has to be understood
afterwards from what it reported. Right now it reports a great deal — logs
in several styles, ad-hoc metrics, degradation records — but there is no
dictionary, so there is no answer to "what are all the values this system
reports, what do they mean, and what counts as bad".

Three things this provides that logs and counters do not:

1. **Declared limits, evaluated on write.** A channel says what yellow
   and red mean for it. Crossing a limit is a state *transition*, not a
   threshold check somebody remembered to write at the read site. The
   transition is the event; a value that has been red for an hour and a
   value that just went red are different situations and the log conflates
   them.
2. **Severity that means something operational.** F Prime's ladder —
   DIAGNOSTIC, ACTIVITY_LO/HI, WARNING_LO/HI, FATAL — is about what a
   ground team should DO, not about how the author felt. WARNING_HI means
   "a human should look before the next pass". FATAL means "the vehicle
   will safe itself". Mapping to that ladder forces the question.
3. **Format strings with typed arguments.** An event is an ID plus
   arguments, rendered for humans at display time. That makes events
   countable and diffable across runs, which a formatted log line is not.

The report is shaped so an OpenMCT-style client can consume it directly:
domain objects with composition, telemetry values with timestamps, and
limit definitions per channel.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from typing import Any

logger = logging.getLogger("Aura.Telemetry")

#: Retained samples per channel. Enough to see a transition's shape
#: without becoming a database.
DEFAULT_HISTORY = 256

#: Retained events. The tail of the event log is what a post-mortem reads.
EVENT_LOG_CAPACITY = 4096


class EventSeverity(IntEnum):
    """What a ground team should DO, not how the author felt."""

    #: Engineering detail. Nobody is paged; useful in a post-mortem.
    DIAGNOSTIC = 0
    #: Normal activity worth recording. The system did a thing on purpose.
    ACTIVITY_LO = 1
    #: Significant normal activity — a mode change, a plan committed.
    ACTIVITY_HI = 2
    #: Something is off but the system is handling it.
    WARNING_LO = 3
    #: A human should look before this gets worse.
    WARNING_HI = 4
    #: The system will protect itself: shed, safe, or restart.
    FATAL = 5

    @property
    def label(self) -> str:
        return self.name


class ChannelType(StrEnum):
    FLOAT = "float"
    INT = "int"
    BOOL = "bool"
    STRING = "string"
    ENUM = "enum"


class LimitState(StrEnum):
    """OpenMCT's limit vocabulary. Order matters for escalation."""

    NOMINAL = "nominal"
    YELLOW_LOW = "yellow_low"
    YELLOW_HIGH = "yellow_high"
    RED_LOW = "red_low"
    RED_HIGH = "red_high"
    STALE = "stale"

    @property
    def is_violation(self) -> bool:
        return self is not LimitState.NOMINAL and self is not LimitState.STALE

    @property
    def is_red(self) -> bool:
        return self in (LimitState.RED_LOW, LimitState.RED_HIGH)


@dataclass(frozen=True)
class Limits:
    yellow_low: float | None = None
    yellow_high: float | None = None
    red_low: float | None = None
    red_high: float | None = None

    def evaluate(self, value: float) -> LimitState:
        if self.red_low is not None and value <= self.red_low:
            return LimitState.RED_LOW
        if self.red_high is not None and value >= self.red_high:
            return LimitState.RED_HIGH
        if self.yellow_low is not None and value <= self.yellow_low:
            return LimitState.YELLOW_LOW
        if self.yellow_high is not None and value >= self.yellow_high:
            return LimitState.YELLOW_HIGH
        return LimitState.NOMINAL

    def coherent(self) -> list[str]:
        """Limits that cannot both be true describe a channel nobody checked."""
        problems: list[str] = []
        if (
            self.red_low is not None
            and self.yellow_low is not None
            and self.red_low > self.yellow_low
        ):
            problems.append("red_low is above yellow_low; yellow can never be reached")
        if (
            self.red_high is not None
            and self.yellow_high is not None
            and self.red_high < self.yellow_high
        ):
            problems.append("red_high is below yellow_high; yellow can never be reached")
        low = self.red_low if self.red_low is not None else self.yellow_low
        high = self.red_high if self.red_high is not None else self.yellow_high
        if low is not None and high is not None and low >= high:
            problems.append("the low limit is at or above the high limit")
        return problems

    def to_dict(self) -> dict[str, Any]:
        return {
            "yellow_low": self.yellow_low,
            "yellow_high": self.yellow_high,
            "red_low": self.red_low,
            "red_high": self.red_high,
        }


@dataclass(frozen=True)
class ChannelSpec:
    identifier: int
    name: str
    type: ChannelType
    unit: str
    description: str
    owner: str
    limits: Limits = field(default_factory=Limits)
    #: A channel silent for longer than this reads STALE, not nominal.
    stale_after_s: float = 120.0
    enum_labels: tuple[str, ...] = ()
    group: str = "general"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.identifier,
            "name": self.name,
            "type": str(self.type),
            "unit": self.unit,
            "description": self.description,
            "owner": self.owner,
            "limits": self.limits.to_dict(),
            "stale_after_s": self.stale_after_s,
            "enum_labels": list(self.enum_labels),
            "group": self.group,
        }


@dataclass(frozen=True)
class Sample:
    at: float
    value: Any
    state: LimitState

    def to_dict(self) -> dict[str, Any]:
        return {"timestamp": self.at, "value": self.value, "limit": str(self.state)}


@dataclass(frozen=True)
class Event:
    identifier: int
    name: str
    severity: EventSeverity
    at: float
    args: dict[str, Any]
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.identifier,
            "name": self.name,
            "severity": self.severity.label,
            "timestamp": self.at,
            "args": dict(self.args),
            "text": self.text,
        }


@dataclass(frozen=True)
class EventSpec:
    identifier: int
    name: str
    severity: EventSeverity
    format_string: str
    description: str
    owner: str

    def render(self, args: dict[str, Any]) -> str:
        try:
            return self.format_string.format(**args)
        except (KeyError, IndexError, ValueError) as exc:
            return f"{self.name}({args}) [format error: {exc}]"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.identifier,
            "name": self.name,
            "severity": self.severity.label,
            "format": self.format_string,
            "description": self.description,
            "owner": self.owner,
        }


@dataclass
class _Channel:
    spec: ChannelSpec
    history: deque[Sample]
    state: LimitState = LimitState.NOMINAL
    state_since: float = 0.0
    writes: int = 0
    violations: int = 0


class TelemetryDictionary:
    """The channel and event dictionary, and the live values."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._channels: dict[str, _Channel] = {}
        self._by_id: dict[int, str] = {}
        self._events: dict[str, EventSpec] = {}
        self._event_ids: dict[int, str] = {}
        self._log: deque[Event] = deque(maxlen=EVENT_LOG_CAPACITY)
        self._transition_listeners: list[Any] = []
        self.events_emitted = 0

    # ── declaration ───────────────────────────────────────────────────
    def declare_channel(self, spec: ChannelSpec, *, history: int = DEFAULT_HISTORY) -> ChannelSpec:
        problems = spec.limits.coherent()
        if problems:
            raise ValueError(
                f"channel {spec.name!r} has incoherent limits: {'; '.join(problems)}"
            )
        with self._lock:
            existing = self._channels.get(spec.name)
            if existing is not None:
                if existing.spec != spec:
                    raise ValueError(
                        f"channel {spec.name!r} already declared by {existing.spec.owner}; "
                        "a channel has one meaning"
                    )
                return spec
            claimed = self._by_id.get(spec.identifier)
            if claimed is not None and claimed != spec.name:
                raise ValueError(
                    f"channel id {spec.identifier} is already {claimed!r}; ids are the "
                    "contract between the runtime and anything reading it"
                )
            self._channels[spec.name] = _Channel(
                spec=spec, history=deque(maxlen=max(1, history)), state_since=time.time()
            )
            self._by_id[spec.identifier] = spec.name
            return spec

    def declare_event(self, spec: EventSpec) -> EventSpec:
        with self._lock:
            existing = self._events.get(spec.name)
            if existing is not None:
                if existing != spec:
                    raise ValueError(f"event {spec.name!r} already declared differently")
                return spec
            claimed = self._event_ids.get(spec.identifier)
            if claimed is not None and claimed != spec.name:
                raise ValueError(f"event id {spec.identifier} is already {claimed!r}")
            self._events[spec.name] = spec
            self._event_ids[spec.identifier] = spec.name
            return spec

    # ── writing ───────────────────────────────────────────────────────
    def write(self, name: str, value: Any) -> LimitState:
        """Record a channel value and evaluate its limits.

        Returns the resulting limit state. A *transition* into or out of a
        violation emits an event, because "it just went red" and "it has
        been red for an hour" are different situations that a threshold
        check at the read site cannot distinguish.
        """
        now = time.time()
        with self._lock:
            entry = self._channels.get(name)
            if entry is None:
                logger.debug("telemetry write to undeclared channel %r ignored", name)
                return LimitState.NOMINAL
            state = LimitState.NOMINAL
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                state = entry.spec.limits.evaluate(float(value))
            entry.history.append(Sample(at=now, value=value, state=state))
            entry.writes += 1
            if state.is_violation:
                entry.violations += 1
            previous = entry.state
            transitioned = previous != state
            if transitioned:
                entry.state = state
                entry.state_since = now
            spec = entry.spec

        if transitioned:
            self._announce_transition(spec, previous, state, value)
        return state

    def _announce_transition(
        self, spec: ChannelSpec, previous: LimitState, state: LimitState, value: Any
    ) -> None:
        severity = (
            EventSeverity.WARNING_HI
            if state.is_red
            else EventSeverity.WARNING_LO
            if state.is_violation
            else EventSeverity.ACTIVITY_LO
        )
        self.emit(
            "channel_limit_transition",
            severity=severity,
            channel=spec.name,
            unit=spec.unit,
            value=value,
            previous=str(previous),
            state=str(state),
        )
        with self._lock:
            listeners = list(self._transition_listeners)
        for listener in listeners:
            try:
                listener(spec.name, previous, state, value)
            except Exception:  # noqa: BLE001 — a listener must not block telemetry
                logger.debug("limit transition listener failed", exc_info=True)

    def on_limit_transition(self, listener: Any) -> None:
        with self._lock:
            self._transition_listeners.append(listener)

    def emit(self, name: str, *, severity: EventSeverity | None = None, **args: Any) -> Event:
        """Emit an event report. Undeclared events still record, at DIAGNOSTIC."""
        with self._lock:
            spec = self._events.get(name)
        if spec is None:
            spec = EventSpec(
                identifier=0,
                name=name,
                severity=severity or EventSeverity.DIAGNOSTIC,
                format_string=name + " {args}",
                description="(undeclared event)",
                owner="unknown",
            )
            text = f"{name} {args}"
        else:
            text = spec.render(args)
        event = Event(
            identifier=spec.identifier,
            name=name,
            severity=severity if severity is not None else spec.severity,
            at=time.time(),
            args=dict(args),
            text=text,
        )
        with self._lock:
            self._log.append(event)
            self.events_emitted += 1

        level = {
            EventSeverity.DIAGNOSTIC: logging.DEBUG,
            EventSeverity.ACTIVITY_LO: logging.DEBUG,
            EventSeverity.ACTIVITY_HI: logging.INFO,
            EventSeverity.WARNING_LO: logging.INFO,
            EventSeverity.WARNING_HI: logging.WARNING,
            EventSeverity.FATAL: logging.CRITICAL,
        }[event.severity]
        logger.log(level, "📡 EVR[%s] %s", event.severity.label, event.text)
        return event

    # ── reading ───────────────────────────────────────────────────────
    def value(self, name: str) -> Sample | None:
        with self._lock:
            entry = self._channels.get(name)
            if entry is None or not entry.history:
                return None
            return entry.history[-1]

    def spec(self, name: str) -> ChannelSpec | None:
        """The channel's declaration, or ``None`` when it has none.

        Distinguishing "undeclared" from "declared and nominal" is the
        whole point: :meth:`state` answers ``NOMINAL`` for a name it has
        never heard of, so a caller that gates on freshness must be able to
        ask whether the channel exists at all. Without this, a misspelled
        channel name reads as permanently healthy.
        """
        with self._lock:
            entry = self._channels.get(name)
            return entry.spec if entry is not None else None

    def is_declared(self, name: str) -> bool:
        with self._lock:
            return name in self._channels

    def state(self, name: str) -> LimitState:
        """Current limit state, with staleness applied."""
        with self._lock:
            entry = self._channels.get(name)
            if entry is None:
                return LimitState.NOMINAL
            if not entry.history:
                return LimitState.STALE
            age = time.time() - entry.history[-1].at
            if age > entry.spec.stale_after_s:
                return LimitState.STALE
            return entry.state

    def history(self, name: str, *, limit: int = 64) -> list[Sample]:
        with self._lock:
            entry = self._channels.get(name)
            return list(entry.history)[-limit:] if entry else []

    def events(self, *, min_severity: EventSeverity = EventSeverity.DIAGNOSTIC, limit: int = 64) -> list[Event]:
        with self._lock:
            return [e for e in self._log if e.severity >= min_severity][-limit:]

    def violations(self) -> list[dict[str, Any]]:
        """Every channel currently out of limits or stale."""
        out: list[dict[str, Any]] = []
        with self._lock:
            names = list(self._channels)
        for name in names:
            state = self.state(name)
            if state is LimitState.NOMINAL:
                continue
            with self._lock:
                entry = self._channels[name]
                since = entry.state_since
                spec = entry.spec
            sample = self.value(name)
            out.append(
                {
                    "channel": name,
                    "id": spec.identifier,
                    "state": str(state),
                    "value": sample.value if sample else None,
                    "unit": spec.unit,
                    "for_s": round(time.time() - since, 1),
                    "limits": spec.limits.to_dict(),
                    "owner": spec.owner,
                }
            )
        return sorted(out, key=lambda entry: -entry["for_s"])

    # ── the dictionary itself ─────────────────────────────────────────
    def dictionary(self) -> dict[str, Any]:
        """The artifact a ground system loads. Ids are the contract."""
        with self._lock:
            channels = [c.spec.to_dict() for c in self._channels.values()]
            events = [e.to_dict() for e in self._events.values()]
        return {
            "version": 1,
            "channels": sorted(channels, key=lambda c: c["id"]),
            "events": sorted(events, key=lambda e: e["id"]),
            "groups": sorted({c["group"] for c in channels}),
        }

    def domain_objects(self) -> list[dict[str, Any]]:
        """OpenMCT-shaped tree: groups compose channels.

        A client that understands domain objects, composition, and
        telemetry metadata can render this without knowing anything about
        Aura specifically.
        """
        with self._lock:
            specs = [c.spec for c in self._channels.values()]
        groups: dict[str, list[str]] = {}
        for spec in specs:
            groups.setdefault(spec.group, []).append(f"aura.telemetry:{spec.name}")

        objects: list[dict[str, Any]] = [
            {
                "identifier": {"namespace": "aura.telemetry", "key": "root"},
                "name": "Aura Runtime",
                "type": "folder",
                "composition": [
                    {"namespace": "aura.telemetry", "key": f"group:{group}"} for group in sorted(groups)
                ],
            }
        ]
        for group, members in sorted(groups.items()):
            objects.append(
                {
                    "identifier": {"namespace": "aura.telemetry", "key": f"group:{group}"},
                    "name": group,
                    "type": "folder",
                    "composition": [
                        {"namespace": "aura.telemetry", "key": key.split(":", 1)[1]}
                        for key in members
                    ],
                }
            )
        for spec in specs:
            objects.append(
                {
                    "identifier": {"namespace": "aura.telemetry", "key": spec.name},
                    "name": spec.name,
                    "type": "aura.telemetry-point",
                    "telemetry": {
                        "values": [
                            {
                                "key": "utc",
                                "source": "timestamp",
                                "name": "Time",
                                "format": "utc",
                                "hints": {"domain": 1},
                            },
                            {
                                "key": "value",
                                "name": spec.description or spec.name,
                                "unit": spec.unit,
                                "hints": {"range": 1},
                                "limits": spec.limits.to_dict(),
                            },
                        ]
                    },
                }
            )
        return objects

    def report(self) -> dict[str, Any]:
        with self._lock:
            channel_count = len(self._channels)
            event_count = len(self._events)
            entries = [(name, c) for name, c in self._channels.items()]
        return {
            "channels": channel_count,
            "declared_events": event_count,
            "events_emitted": self.events_emitted,
            "violations": self.violations(),
            "recent_events": [
                e.to_dict() for e in self.events(min_severity=EventSeverity.WARNING_LO, limit=8)
            ],
            "silent_channels": [
                name for name, entry in entries if not entry.history
            ],
            "busiest": sorted(
                ((name, entry.writes) for name, entry in entries),
                key=lambda kv: -kv[1],
            )[:8],
        }

    def reset_for_test(self) -> None:
        with self._lock:
            self._channels.clear()
            self._by_id.clear()
            self._events.clear()
            self._event_ids.clear()
            self._log.clear()
            self._transition_listeners.clear()
            self.events_emitted = 0


_DICTIONARY = TelemetryDictionary()


def get_telemetry() -> TelemetryDictionary:
    return _DICTIONARY


def channel(
    identifier: int,
    name: str,
    *,
    type: ChannelType = ChannelType.FLOAT,  # noqa: A002 — mirrors the spec field
    unit: str = "",
    description: str = "",
    owner: str = "unknown",
    yellow_low: float | None = None,
    yellow_high: float | None = None,
    red_low: float | None = None,
    red_high: float | None = None,
    stale_after_s: float = 120.0,
    group: str = "general",
    enum_labels: tuple[str, ...] = (),
) -> ChannelSpec:
    """Declare a telemetry channel with its limits."""
    return _DICTIONARY.declare_channel(
        ChannelSpec(
            identifier=identifier,
            name=name,
            type=type,
            unit=unit,
            description=description,
            owner=owner,
            limits=Limits(
                yellow_low=yellow_low,
                yellow_high=yellow_high,
                red_low=red_low,
                red_high=red_high,
            ),
            stale_after_s=stale_after_s,
            group=group,
            enum_labels=enum_labels,
        )
    )


def event(
    identifier: int,
    name: str,
    *,
    severity: EventSeverity,
    format_string: str,
    description: str = "",
    owner: str = "unknown",
) -> EventSpec:
    return _DICTIONARY.declare_event(
        EventSpec(
            identifier=identifier,
            name=name,
            severity=severity,
            format_string=format_string,
            description=description,
            owner=owner,
        )
    )


def channel_value(name: str) -> Sample | None:
    """The last sample on a channel, or None where nothing has written it."""
    return _DICTIONARY.value(name)


def write(name: str, value: Any) -> LimitState:
    return _DICTIONARY.write(name, value)


def emit_event(name: str, *, severity: EventSeverity | None = None, **args: Any) -> Event:
    return _DICTIONARY.emit(name, severity=severity, **args)


def telemetry_report() -> dict[str, Any]:
    return _DICTIONARY.report()


def reset_telemetry_for_test() -> None:
    _DICTIONARY.reset_for_test()


__all__ = [
    "ChannelSpec",
    "ChannelType",
    "Event",
    "EventSeverity",
    "EventSpec",
    "LimitState",
    "Limits",
    "Sample",
    "TelemetryDictionary",
    "channel",
    "emit_event",
    "event",
    "get_telemetry",
    "reset_telemetry_for_test",
    "telemetry_report",
    "write",
]
