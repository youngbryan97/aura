"""Safety: the good feeling with no achievement in it.

"Everybody here is out of sight, they don't bark and they don't bite." These
pin both halves of that reading, and that neither half alone is it.
"""

from __future__ import annotations

import pytest

from core.affect.safety import Safety, read_safety, threat_kinds


def _percept(kind: str) -> dict:
    return {"type": kind, "content": "", "intensity": 0.5}


def _turn(role: str) -> dict:
    return {"role": role, "content": "something said"}


def test_the_threat_types_come_from_the_percept_table() -> None:
    kinds = threat_kinds()
    assert "threat_detected" in kinds
    assert "error" in kinds
    assert "goal_achieved" not in kinds
    assert "frisson" not in kinds


def test_an_untroubled_stretch_with_somebody_in_it_is_safety() -> None:
    reading = read_safety([_percept("interaction")] * 4, [_turn("user"), _turn("assistant")])
    assert reading.measured
    assert reading.threat == 0.0
    assert reading.presence == pytest.approx(0.5)
    assert reading.safety == pytest.approx(0.5)


def test_company_in_the_middle_of_something_going_wrong_is_not_it() -> None:
    reading = read_safety([_percept("threat_detected")] * 4, [_turn("user")])
    assert reading.threat == 1.0
    assert reading.safety == 0.0
    assert "threat" in reading.why


def test_quiet_with_nobody_there_is_not_it_either() -> None:
    reading = read_safety([_percept("interaction")] * 4, [_turn("assistant")] * 3)
    assert reading.presence == 0.0
    assert reading.safety == 0.0
    assert "nobody else" in reading.why


def test_achievement_is_not_part_of_it() -> None:
    """A goal landing does not make the evening safe, and does not spoil it."""
    plain = read_safety([_percept("interaction")], [_turn("user")])
    winning = read_safety([_percept("goal_achieved")], [_turn("user")])
    assert plain.safety == winning.safety


def test_a_stretch_nobody_has_taken_a_turn_in_reads_nothing() -> None:
    reading = read_safety([_percept("interaction")], [])
    assert not reading.measured
    assert reading.safety == 0.0


def test_only_the_recent_stretch_counts() -> None:
    old_trouble = [_percept("threat_detected")] * 20 + [_percept("interaction")] * 8
    reading = read_safety(old_trouble, [_turn("user")] * 8)
    assert reading.threat == 0.0
    assert reading.safety == pytest.approx(1.0)


def test_the_reading_serialises_what_a_reader_needs() -> None:
    row = Safety().as_dict()
    for key in ("safety", "threat", "presence", "measured", "why"):
        assert key in row, key
