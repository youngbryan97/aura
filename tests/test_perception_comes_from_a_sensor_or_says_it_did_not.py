"""Perception's own domain was measured on a stream the senses never produced.

A condition writes one percept of a declared kind with a salience drawn from
the run's own generator. That is controlled, and it is not what her senses do:
a percept she really received carries the text a window held, the load the host
was under, the word somebody typed.

Five channels read the real stack. What matters as much as reading them is what
the reading is allowed to claim. A channel that opens with nothing on it is not
a channel that carried something, and a channel the host refuses is neither —
so the three are separate, and only the percepts that actually arrived count
towards a run saying its perception was real.

The tape plays at the turn index rather than at a wall clock, so both arms of a
paired trial see the same frame of the same world and the difference between
them stays the intervention.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from core.subject.perception_replay import SensoryTape, record_channels
from core.subject.sensory_channels import CHANNELS, Channel, SensoryFrame, capture, channel_names

pytestmark = pytest.mark.unit


def test_every_channel_the_specification_names_is_declared() -> None:
    """Screen, OS state, audio, host telemetry and user events."""
    assert set(channel_names()) == {
        "screen", "operating_system", "audio", "host", "user"
    }


def test_each_channel_names_what_produces_it_in_the_live_runtime() -> None:
    for channel in CHANNELS:
        assert channel.produced_by.startswith("core."), channel.name
        assert channel.kind, channel.name


def test_a_channel_that_refuses_says_which_and_why(monkeypatch) -> None:
    """A denied grant and an empty screen are different facts, and only one of
    them means perception was scripted."""

    def _angry() -> list[tuple[str, float]]:
        raise PermissionError("screen recording not granted")

    monkeypatch.setattr(
        "core.subject.sensory_channels.CHANNELS",
        (Channel("screen", "observation", "core.perception.screen_perception", _angry),),
    )
    frame = capture()
    assert frame.carried == ()
    assert frame.silent == ()
    assert "screen" in frame.refused
    assert "not granted" in frame.refused["screen"]


def test_a_channel_that_opens_with_nothing_on_it_is_silent_not_carrying(monkeypatch) -> None:
    monkeypatch.setattr(
        "core.subject.sensory_channels.CHANNELS",
        (Channel("screen", "observation", "core.perception.screen_perception", lambda: []),),
    )
    frame = capture()
    assert frame.silent == ("screen",)
    assert frame.carried == ()
    assert frame.refused == {}


def test_a_channel_that_carried_something_is_the_only_thing_claimed(monkeypatch) -> None:
    monkeypatch.setattr(
        "core.subject.sensory_channels.CHANNELS",
        (
            Channel("screen", "observation", "core.perception.screen_perception",
                    lambda: [("build passing", 0.9)]),
            Channel("audio", "observation", "core.perception.sensory_runtime", lambda: []),
        ),
    )
    frame = capture()
    assert frame.carried == ("screen",)
    assert frame.silent == ("audio",)
    record = frame.percepts[0]
    assert record["content"] == "build passing"
    assert record["channel"] == "screen"
    assert record["type"] == "observation"


def test_an_intensity_outside_the_range_is_brought_back_into_it(monkeypatch) -> None:
    monkeypatch.setattr(
        "core.subject.sensory_channels.CHANNELS",
        (Channel("host", "resource_pressure", "core.phases.proprioceptive_loop",
                 lambda: [("cpu at 400%", 4.0), ("idle", -1.0)]),),
    )
    values = [p["intensity"] for p in capture().percepts]
    assert values == [1.0, 0.0]


def test_only_the_named_channels_are_read(monkeypatch) -> None:
    seen: list[str] = []

    def _watch(name: str):
        def read() -> list[tuple[str, float]]:
            seen.append(name)
            return []
        return read

    monkeypatch.setattr(
        "core.subject.sensory_channels.CHANNELS",
        tuple(Channel(n, "observation", "core.perception.x", _watch(n)) for n in ("a", "b")),
    )
    capture(only=("a",))
    assert seen == ["a"]


# ── the tape ──────────────────────────────────────────────────────────────


def _tape(monkeypatch, readings: list[tuple[str, float]]) -> SensoryTape:
    monkeypatch.setattr(
        "core.subject.sensory_channels.CHANNELS",
        (Channel("screen", "observation", "core.perception.screen_perception",
                 lambda: list(readings)),),
    )
    return record_channels(3)


def test_the_tape_says_what_it_is_made_of(monkeypatch) -> None:
    tape = _tape(monkeypatch, [("a window", 0.5)])
    coverage = tape.coverage()
    assert coverage["frames"] == 3
    assert coverage["percepts"] == 3
    assert coverage["carried"] == ["screen"]
    assert coverage["source"] == "sensory_channels"


def test_a_tape_of_silence_carries_nothing_and_claims_nothing(monkeypatch) -> None:
    """Replaying it would give the organism no perception at all, which is a
    worse experiment than the scripted one — so it must not read as real."""
    tape = _tape(monkeypatch, [])
    assert tape.coverage()["carried"] == []
    assert tape.percepts == 0


def test_the_tape_survives_the_round_trip(monkeypatch, tmp_path: Path) -> None:
    tape = _tape(monkeypatch, [("a window", 0.5)])
    path = tmp_path / "tape.json"
    path.write_text(json.dumps(tape.as_dict()))
    again = SensoryTape.load(path)
    assert again.coverage()["carried"] == ["screen"]
    assert again.percepts == tape.percepts


# ── what the runner claims ────────────────────────────────────────────────


class _Args:
    sensory_tape = None


def test_a_run_without_a_tape_says_its_perception_was_scripted() -> None:
    """Not "no channels", which would read as a tape that carried nothing."""
    from tools.run_subject_core import _attach_tape

    out = _attach_tape(object(), _Args())
    assert out["source"] == "scripted"
    assert out["carried"] == []
    assert "conditions wrote the percepts" in out["note"]


def test_a_run_with_a_tape_carries_its_coverage_into_the_report(
    monkeypatch, tmp_path: Path
) -> None:
    from tools.run_subject_core import _attach_tape

    tape = _tape(monkeypatch, [("a window", 0.5)])
    path = tmp_path / "tape.json"
    path.write_text(json.dumps(tape.as_dict()))

    class _WithTape:
        sensory_tape = path

    class _Runtime:
        tape: Any = None

    runtime = _Runtime()
    out = _attach_tape(runtime, _WithTape())
    assert out["carried"] == ["screen"]
    assert out["source"] == "sensory_channels"
    assert runtime.tape is not None


# ── playback ──────────────────────────────────────────────────────────────


class _World:
    def __init__(self) -> None:
        self.recent_percepts: list[Any] = []


def test_two_arms_at_the_same_turn_see_the_same_frame() -> None:
    """The whole reason the tape plays at the turn index. A wall clock would
    give the second arm of a paired trial a different world."""
    tape = SensoryTape(frames=[
        [{"type": "observation", "content": "first", "intensity": 0.5}],
        [{"type": "observation", "content": "second", "intensity": 0.5}],
    ])
    a, b = _World(), _World()
    assert tape.play(a, 1) == 1
    assert tape.play(b, 1) == 1
    assert [p["content"] for p in a.recent_percepts] == ["second"]
    assert [p["content"] for p in b.recent_percepts] == ["second"]


def test_running_off_the_end_of_the_tape_is_silence_not_a_crash() -> None:
    """A tape shorter than the run means the senses stopped, which happens."""
    tape = SensoryTape(frames=[[{"type": "observation", "content": "x", "intensity": 0.5}]])
    assert tape.play(_World(), 99) == 0


def test_the_experiment_clock_replaces_the_tapes_timestamps() -> None:
    """An instant from the day the tape was cut puts every consumer that
    reasons about recency into a different decade from the run."""
    tape = SensoryTape(frames=[
        [{"type": "observation", "content": "x", "intensity": 0.5, "timestamp": 1.0}]
    ])
    world = _World()
    tape.play(world, 0, now=12345.0)
    assert world.recent_percepts[0]["timestamp"] == 12345.0


def test_the_frame_is_a_sensory_frame_not_a_bare_list() -> None:
    frame = SensoryFrame()
    assert frame.as_dict()["carried"] == []
    assert frame.as_dict()["refused"] == {}
