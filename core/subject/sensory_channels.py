"""The channels a real percept can arrive on, and what each one yields here.

Perception's own domain is measured on a stream the perception stack never
produced. A condition writes one percept of a declared kind with a salience
drawn from the run's own generator: controlled, and nothing like what her
senses do. A percept she really received carries the text a window held, the
load the host was under, the word somebody typed.

So this reads the real stack. Five channels — screen, operating system, audio,
host telemetry and user events — each with the module that produces it and the
percept kind it arrives as. A channel the host refuses says which channel and
why, and it is never quietly replaced by a scripted one: a run that claims real
perception on a channel it could not open has claimed the scripted stream under
a different name.

    from core.subject.sensory_channels import capture

    frame = capture()            # every channel that will open
    frame.coverage["screen"]     # "live", or the reason it is not

Nothing here asks for a permission the host has not already granted, and
nothing starts a capture daemon. A channel whose reader needs a grant this
process does not hold reports `refused` and the run says so.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["CHANNELS", "Channel", "SensoryFrame", "capture", "channel_names"]


@dataclass(frozen=True)
class Channel:
    """One way the world reaches her, and how to read it here."""

    name: str
    #: The percept kind this arrives as, which is what the affect phase keys on.
    kind: str
    #: What produces it in the live runtime, for the record.
    produced_by: str
    read: Callable[[], list[tuple[str, float]]]


@dataclass
class SensoryFrame:
    """One frame of real perception, and what each channel had to say."""

    percepts: list[dict[str, Any]] = field(default_factory=list)
    coverage: dict[str, str] = field(default_factory=dict)

    @property
    def carried(self) -> tuple[str, ...]:
        """Channels that opened and had something on them."""
        return tuple(sorted(n for n, why in self.coverage.items() if why == "live"))

    @property
    def silent(self) -> tuple[str, ...]:
        """Channels that opened with nothing on them.

        Separate from both of the others on purpose. A channel that opens and
        carries nothing every frame contributed no real percept, so a run must
        not count it towards a claim that perception was real — and it is not a
        refusal either, because an empty screen is a reading.
        """
        return tuple(sorted(n for n, why in self.coverage.items() if why == "silent"))

    @property
    def refused(self) -> dict[str, str]:
        return {
            n: why for n, why in self.coverage.items() if why not in ("live", "silent")
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "percepts": list(self.percepts),
            "coverage": dict(self.coverage),
            "carried": list(self.carried),
            "silent": list(self.silent),
            "refused": dict(self.refused),
        }


def _observations(*kinds: Any) -> list[tuple[str, float]]:
    """Whatever perception already captured, of these kinds.

    Read from the observation record rather than by taking a capture: a battery
    that screenshots the host changes the world it is measuring, and the grant
    belongs to the live instance.
    """
    from core.perception.observation_evidence import get_observation_memory

    wanted = set(kinds)
    out: list[tuple[str, float]] = []
    for item in get_observation_memory().recent():
        if item.kind not in wanted or item.is_empty:
            continue
        # Older captures carry less: an observation from a minute ago is a
        # weaker percept than the one that just arrived, and the stream prices
        # its bid off intensity.
        age = max(0.0, time.time() - float(item.at))
        out.append((item.capture[:400], round(max(0.1, 1.0 - age / 120.0), 3)))
    return out


def _screen() -> list[tuple[str, float]]:
    from core.perception.observation_evidence import ObservationKind

    return _observations(ObservationKind.SCREEN_TEXT, ObservationKind.WINDOW_TREE)


def _operating_system() -> list[tuple[str, float]]:
    """What the machine is doing: files, logs, terminals, sockets, power.

    The ambient stream's latest frame, not a fresh collection. Sampling it here
    would put this process's own reading into the frame the run is measuring.
    """
    from core.perception.ambient_developer_stream import get_ambient_developer_stream

    frame = get_ambient_developer_stream().latest_frame
    if frame is None:
        return []
    out: list[tuple[str, float]] = []
    if frame.summary:
        out.append((frame.summary[:400], 0.5))
    for event in frame.resource_interrupts:
        out.append((f"{event.kind}: {_bounded(event)}", 0.9))
    for event in frame.network_events:
        out.append((f"{event.kind}: {event.count}", 0.4))
    for event in frame.terminal_events:
        out.append((_bounded(event)[:400], 0.6))
    return out


def _bounded(event: Any) -> str:
    for name in ("detail", "line", "text", "summary", "path", "message"):
        value = getattr(event, name, "")
        if value:
            return str(value)
    return str(getattr(event, "kind", "event"))


def _audio() -> list[tuple[str, float]]:
    from core.perception.observation_evidence import ObservationKind

    return _observations(ObservationKind.AUDIO)


def _host() -> list[tuple[str, float]]:
    """Her own body: load, thermals, memory. Always available, and the reason
    this list is never empty even on a machine that grants nothing."""
    import psutil

    load = float(psutil.cpu_percent(interval=None)) / 100.0
    memory = psutil.virtual_memory()
    return [
        (f"cpu at {load:.0%}", min(1.0, load)),
        (
            f"memory {memory.percent:.0f}% used, {memory.available / 1e9:.1f}GB free",
            min(1.0, float(memory.percent) / 100.0),
        ),
    ]


def _user() -> list[tuple[str, float]]:
    """What somebody did: what they put on the clipboard, and what they showed
    her a picture of."""
    from core.perception.observation_evidence import ObservationKind

    return _observations(ObservationKind.CLIPBOARD, ObservationKind.SCREEN_IMAGE)


#: Every channel, declared here rather than discovered, so a run that recorded
#: four of them says which one is missing.
CHANNELS: tuple[Channel, ...] = (
    Channel("screen", "observation", "core.perception.screen_perception", _screen),
    Channel("operating_system", "discovery", "core.perception.ambient_developer_stream", _operating_system),
    Channel("audio", "observation", "core.perception.sensory_runtime", _audio),
    Channel("host", "resource_pressure", "core.phases.proprioceptive_loop", _host),
    Channel("user", "interaction", "core.perception.ambient_presence", _user),
)


def channel_names() -> tuple[str, ...]:
    return tuple(channel.name for channel in CHANNELS)


def capture(*, only: tuple[str, ...] | None = None) -> SensoryFrame:
    """Read every channel once. A channel that will not open says why.

    A reader that raises is a refusal with a reason, not an empty channel: a
    host that denies screen access and a screen with nothing on it are
    different facts, and only one of them means perception was scripted.
    """
    wanted = set(only) if only else None
    frame = SensoryFrame()
    for channel in CHANNELS:
        if wanted is not None and channel.name not in wanted:
            continue
        try:
            readings = channel.read()
        except ImportError as exc:
            frame.coverage[channel.name] = f"unavailable: {exc}"
            continue
        except Exception as exc:  # noqa: BLE001 - the reason is the result
            frame.coverage[channel.name] = f"refused: {type(exc).__name__}: {exc}"
            logger.debug("sensory channel %s refused: %s", channel.name, exc)
            continue
        if not readings:
            frame.coverage[channel.name] = "silent"
            continue
        frame.coverage[channel.name] = "live"
        for content, intensity in readings:
            frame.percepts.append({
                "type": channel.kind,
                "content": content,
                "intensity": round(float(max(0.0, min(1.0, intensity))), 3),
                "source": channel.name,
                "channel": channel.name,
                "produced_by": channel.produced_by,
            })
    return frame
