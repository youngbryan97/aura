"""Two arms must see the same world, and it should be a world she really saw.

The battery's conditions script perception: one percept of a declared kind with
a salience drawn from the run's own generator. That is controlled, and it is
not what her senses do — a percept she really received carries the content a
screen held, the words somebody typed, the load the host was under, and a
scripted one carries none of it. Perception's own domain was measured on a
stream the perception stack never produced.

A tape is the other half: percepts as they arrived, handed back frame by frame,
so the world two arms see is identical and the difference between them stays
the intervention.
"""

from __future__ import annotations

from pathlib import Path

from core.state.aura_state import AuraState
from core.state.percepts import emit_percept
from core.subject.perception_replay import SensoryTape


def _world_with(*percepts: tuple[str, str, float]):
    state = AuraState.default()
    for kind, content, intensity in percepts:
        emit_percept(state.world, kind, content=content, intensity=intensity, source="screen")
    return state.world


def test_a_tape_keeps_what_arrived() -> None:
    world = _world_with(("interaction", "he asked about the plan", 0.8))
    tape = SensoryTape.record(world)
    assert len(tape) == 1
    assert tape.percepts == 1


def test_two_arms_replaying_one_frame_see_the_same_world() -> None:
    tape = SensoryTape.record(
        _world_with(
            ("interaction", "he asked about the plan", 0.8),
            ("discovery", "the file was already there", 0.3),
        )
    )
    left, right = AuraState.default(), AuraState.default()
    assert tape.play(left.world, 0, now=1000.0) == 2
    assert tape.play(right.world, 0, now=1000.0) == 2

    def seen(state):
        return [
            (p.get("type"), p.get("content"), p.get("intensity"), p.get("source"))
            for p in state.world.recent_percepts
        ]

    assert seen(left) == seen(right)


def test_a_replayed_percept_carries_what_a_live_one_does() -> None:
    """Indistinguishable downstream, or replay is a different experiment."""
    tape = SensoryTape.record(_world_with(("discovery", "a note from yesterday", 0.6)))
    state = AuraState.default()
    tape.play(state.world, 0, now=42.0)
    record = state.world.recent_percepts[-1]
    assert record["type"] == "discovery"
    assert record["content"] == "a note from yesterday"
    assert record["intensity"] == 0.6
    assert record["source"] == "screen"
    assert record["timestamp"] == 42.0, (
        "an instant from the day the tape was cut would put every consumer "
        "that reasons about recency into a different decade from the run"
    )


def test_a_frame_past_the_end_is_silence_rather_than_a_crash() -> None:
    tape = SensoryTape.record(_world_with(("interaction", "x", 0.5)))
    state = AuraState.default()
    assert tape.play(state.world, 9, now=1.0) == 0
    assert tape.play(state.world, -1, now=1.0) == 0


def test_a_tape_survives_the_round_trip(tmp_path: Path) -> None:
    tape = SensoryTape.record(
        _world_with(("interaction", "keep this", 0.9), ("error", "and this", 0.2))
    )
    path = tape.save(tmp_path / "tape.json")
    again = SensoryTape.load(path)
    assert len(again) == len(tape)
    assert again.frames == tape.frames


def test_record_stream_captures_one_frame_at_a_time() -> None:
    from core.subject.perception_replay import record_stream

    state = AuraState.default()
    counter = {"n": 0}

    def step() -> None:
        counter["n"] += 1
        emit_percept(state.world, "interaction", content=f"turn {counter['n']}", intensity=0.5)

    tape = record_stream(state.world, 3, step)
    assert len(tape) == 3
    assert counter["n"] == 3
