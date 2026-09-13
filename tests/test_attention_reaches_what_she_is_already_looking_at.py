"""Biased competition, for percepts that were already in the stream.

What wins the global workspace biases the perceptual systems. `_attend`
implements the onset half of that: an arriving percept sharing content with the
broadcast comes in at a higher salience. The other half was missing, and it is
the half that carries a measurement.

Attention is not an onset effect. A shift of attention should reach what she is
already looking at, not only what happens to arrive in the same tick. Without
the second half the workspace could only reach perception by coincidence, and
in run_019 the whole G-to-P channel measured 0.17 against a bar of 0.30 and
replicated in three conditions of eight — a channel firing on whether a percept
and a broadcast landed together.

The gain carries no constant. It closes the gap to full salience in proportion
to the overlap and to how strongly the broadcast won, so a percept with nothing
in common comes back exactly as it was, and nothing can be raised past one.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.state.percepts import emit_percept, reweight_stream

pytestmark = pytest.mark.unit


def _world(*contents: str, intensity: float = 0.2):
    world = SimpleNamespace(recent_percepts=[])
    for content in contents:
        emit_percept(world, "interaction", content=content, intensity=intensity, source="chat")
    return world


def _salience(world):
    return [round(float(p["salience"]), 6) for p in world.recent_percepts]


def test_what_is_attended_raises_what_is_already_in_the_stream() -> None:
    world = _world("Bryan sent a message", "the log holds 12 lines")
    assert reweight_stream(world, {"content": "a message from Bryan arrived", "priority": 0.9}) == 1
    raised, untouched = _salience(world)
    assert raised > 0.2
    assert untouched == 0.2


def test_a_percept_with_nothing_in_common_is_left_exactly_as_it_arrived() -> None:
    world = _world("Bryan sent a message")
    before = _salience(world)
    assert reweight_stream(world, {"content": "zzzz qqqq wwww", "priority": 1.0}) == 0
    assert _salience(world) == before


def test_a_broadcast_that_did_not_win_changes_nothing() -> None:
    world = _world("Bryan sent a message")
    before = _salience(world)
    assert reweight_stream(world, {"content": "Bryan sent a message", "priority": 0.0}) == 0
    assert _salience(world) == before


def test_the_gain_only_closes_the_gap_and_never_passes_one() -> None:
    world = _world("Bryan sent a message", intensity=0.9)
    reweight_stream(world, {"content": "Bryan sent a message", "priority": 1.0})
    assert _salience(world)[0] <= 1.0


def test_a_stronger_broadcast_raises_it_further() -> None:
    """The channel is graduated, which is what makes it measurable."""
    weak = _world("Bryan sent a message")
    strong = _world("Bryan sent a message")
    reweight_stream(weak, {"content": "Bryan sent a message", "priority": 0.2})
    reweight_stream(strong, {"content": "Bryan sent a message", "priority": 0.9})
    assert _salience(strong)[0] > _salience(weak)[0]


def test_more_overlap_raises_it_further() -> None:
    little = _world("Bryan sent a message about the weather today")
    lots = _world("Bryan sent a message")
    reweight_stream(little, {"content": "Bryan sent a message", "priority": 0.9})
    reweight_stream(lots, {"content": "Bryan sent a message", "priority": 0.9})
    assert _salience(lots)[0] > _salience(little)[0]


def test_an_absent_stream_is_not_an_error() -> None:
    assert reweight_stream(SimpleNamespace(), {"content": "x", "priority": 1.0}) == 0
    assert reweight_stream(None, {"content": "x", "priority": 1.0}) == 0


def test_an_absent_broadcast_is_not_an_error() -> None:
    world = _world("Bryan sent a message")
    assert reweight_stream(world, None) == 0


def test_the_workspace_calls_it_when_it_names_a_winner() -> None:
    """A mechanism nothing calls is a mechanism that cannot fire."""
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1] / "core" / "consciousness" / "workspace_feed.py"
    ).read_text(encoding="utf-8")
    assert "reweight_stream" in source
    assert source.index("attention_focus = ") < source.index("reweight_stream(")
