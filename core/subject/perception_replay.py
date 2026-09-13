"""A recorded sensory stream, replayed identically into matched arms.

The battery's conditions script perception: a condition writes one percept of a
declared kind with a salience drawn from the run's own generator. That is
controlled, and it is not what her senses do. A percept she really received
carries the content a screen held, the words somebody typed, the load the host
was under — and a scripted one carries none of it, so perception's own domain
is measured on a stream the perception stack never produced.

This is the other half: a file of percepts as they actually arrived, and a
reader that hands them back frame by frame. Two arms of a paired trial read the
same file from the same position, so the world they see is identical and the
difference between them stays the intervention.

    from core.subject.perception_replay import SensoryTape

    tape = SensoryTape.record(world)          # while she is living normally
    write_json(directory, "sensory_tape.json", tape.as_dict())
    tape = SensoryTape.load(path)             # in the battery
    tape.play(state.world, frame)             # the same frame, the same world

What is recorded is what `core.state.percepts` already carries, so a replayed
percept is indistinguishable from a live one to everything downstream. The
timestamps are replaced by the experiment's clock on playback: an instant from
the day the tape was cut would put every consumer that reasons about recency
into a different decade from the run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["SensoryTape", "record_channels", "record_stream"]

#: What a percept carries. Anything else on the record is kept as it was.
_REPLACED_ON_PLAY: tuple[str, ...] = ("timestamp",)


@dataclass
class SensoryTape:
    """Percepts as they arrived, in order, with what was true when each did."""

    frames: list[list[dict[str, Any]]] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)

    #: How many frames the tape holds.
    def __len__(self) -> int:
        return len(self.frames)

    @property
    def percepts(self) -> int:
        return sum(len(frame) for frame in self.frames)

    @classmethod
    def record(cls, world: Any, *, notes: dict[str, Any] | None = None) -> SensoryTape:
        """Start a tape from whatever is on the stream right now."""
        tape = cls(notes=dict(notes or {}))
        tape.capture(world)
        return tape

    def capture(self, world: Any) -> int:
        """Append this frame's percepts. Returns how many were on it."""
        stream = list(getattr(world, "recent_percepts", []) or [])
        frame = [dict(item) for item in stream if isinstance(item, dict)]
        self.frames.append(frame)
        return len(frame)

    def play(self, world: Any, frame: int, *, now: float | None = None) -> int:
        """Put the percepts of one frame onto a world. Returns how many.

        Out of range is empty rather than an error: a tape shorter than the run
        means the senses stopped, which is a thing that happens, and a run that
        crashed at the end of a tape would tell you nothing about the organism.
        """
        from core.state.percepts import emit_percept

        if frame < 0 or frame >= len(self.frames):
            return 0
        played = 0
        for record in self.frames[frame]:
            body = {
                key: value
                for key, value in record.items()
                if key not in _REPLACED_ON_PLAY and key not in {"type", "content", "intensity", "salience"}
            }
            emitted = emit_percept(
                world,
                str(record.get("type", "interaction")),
                content=str(record.get("content", "")),
                intensity=float(record.get("intensity", 0.5) or 0.0),
                salience=(
                    None if record.get("salience") is None else float(record["salience"])
                ),
                **body,
            )
            if emitted is not None:
                played += 1
                if now is not None:
                    emitted["timestamp"] = float(now)
        return played

    def as_dict(self) -> dict[str, Any]:
        """The tape as data. Writing it is the archive's job.

        A module that holds a recording and also owns a file is two things, and
        the second one is a call site somebody has to govern. `core.subject.
        archive` already owns every file a run writes:

            write_json(directory, "sensory_tape.json", tape.as_dict())
        """
        return {"notes": dict(self.notes), "frames": [list(f) for f in self.frames]}

    @classmethod
    def from_dict(cls, blob: dict[str, Any]) -> SensoryTape:
        frames = [
            [dict(item) for item in frame if isinstance(item, dict)]
            for frame in blob.get("frames", [])
        ]
        return cls(frames=frames, notes=dict(blob.get("notes", {})))

    @classmethod
    def load(cls, path: Path) -> SensoryTape:
        return cls.from_dict(json.loads(Path(path).read_text()))

    @property
    def carried(self) -> tuple[str, ...]:
        """Channels that put at least one percept on this tape.

        The claim a run is entitled to make. A channel that opened and stayed
        empty is not in here, because it contributed nothing a scripted percept
        did not.
        """
        named = {
            str(record.get("channel", ""))
            for frame in self.frames
            for record in frame
            if record.get("channel")
        }
        return tuple(sorted(named))

    def coverage(self) -> dict[str, Any]:
        """What the tape is made of, for the run's record."""
        return {
            "frames": len(self.frames),
            "percepts": self.percepts,
            "carried": list(self.carried),
            "declared": dict(self.notes.get("coverage", {})),
            "source": self.notes.get("source", "world"),
        }


def record_channels(frames: int, *, wait: Any = None, only: tuple[str, ...] | None = None) -> SensoryTape:
    """A tape cut from the real sensory channels rather than from a world.

    `record_stream` keeps what a world already holds, which is right when the
    organism is living in front of you and wrong when the question is what her
    senses carried. This reads `core.subject.sensory_channels` directly, so the
    tape holds the text a window held and the load the host was under.

    The tape's notes carry which channels opened, which opened with nothing on
    them, and which refused and why. A run that replays this tape can then say
    what its perception was made of instead of asserting that it was real.
    """
    from core.subject.sensory_channels import capture

    tape = SensoryTape(notes={"frames_requested": int(frames), "source": "sensory_channels"})
    coverage: dict[str, dict[str, int]] = {}
    for index in range(max(0, int(frames))):
        if wait is not None and index:
            wait()
        frame = capture(only=only)
        tape.frames.append(frame.percepts)
        for name, why in frame.coverage.items():
            row = coverage.setdefault(name, {"live": 0, "silent": 0, "refused": 0})
            row["live" if why == "live" else "silent" if why == "silent" else "refused"] += 1
            if why not in ("live", "silent"):
                row.setdefault("reason", why)  # type: ignore[arg-type]
    tape.notes["coverage"] = coverage
    tape.notes["carried"] = sorted(n for n, row in coverage.items() if row["live"])
    return tape


def record_stream(world: Any, frames: int, step: Any) -> SensoryTape:
    """Run `step` `frames` times and keep what arrived on each one.

    `step` is whatever advances the organism one frame. Kept general because
    what is being recorded is her senses, not the battery: the same function
    records a desktop session and a harness run.
    """
    tape = SensoryTape(notes={"frames_requested": int(frames)})
    for _ in range(max(0, int(frames))):
        step()
        tape.capture(world)
    return tape
