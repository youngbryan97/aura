"""Two stores for one personality, and the schema recorded the still one.

`S.trait_openness` and its four siblings read 0.0000 on every frame of a probe
— five of the thirty-one columns her self-state could not move. They record
`identity.personality_growth`, which is written in one place, behind a gate at
a bonding level of 0.3. Bonding rises by a ten-thousandth a turn, so the gate
cannot open inside a run, and conscientiousness has no writer on that path at
all.

Her personality does move. `core/brain/personality_engine.py` shifts
`self.traits` every turn from what she has been thinking and how exchanges have
gone. That is the drift the offsets are for.
"""

from __future__ import annotations

from unittest import mock

import pytest

from core.brain.personality_engine import PersonalityEngine
from core.phases.bonding_phase import _PERSONALITY_GROWTH_KEYS, BondingPhase


@pytest.fixture
def engine() -> PersonalityEngine:
    return PersonalityEngine()


def _mirror(engine: PersonalityEngine) -> dict[str, float]:
    growth = dict.fromkeys(_PERSONALITY_GROWTH_KEYS, 0.0)
    with mock.patch(
        "core.brain.personality_engine.get_personality_engine", return_value=engine
    ):
        BondingPhase._mirror_the_drift(growth)
    return growth


def test_the_engine_records_where_she_started(engine):
    """Drift is a difference, so there has to be something to differ from."""
    assert engine._baseline_traits == engine.traits
    assert set(engine._baseline_traits) >= set(_PERSONALITY_GROWTH_KEYS)


def test_a_personality_that_has_not_moved_reports_no_growth(engine):
    assert _mirror(engine) == dict.fromkeys(_PERSONALITY_GROWTH_KEYS, 0.0)


def test_the_drift_the_engine_made_reaches_the_offsets(engine):
    engine.traits["openness"] -= 0.03
    engine.traits["conscientiousness"] += 0.02
    growth = _mirror(engine)
    assert growth["openness"] == pytest.approx(-0.03)
    assert growth["conscientiousness"] == pytest.approx(0.02)
    assert growth["extraversion"] == pytest.approx(0.0)


def test_conscientiousness_can_move_at_last(engine):
    """It is the one the bonding path never wrote under any bonding level."""
    engine.traits["conscientiousness"] += 0.05
    assert _mirror(engine)["conscientiousness"] == pytest.approx(0.05)


def test_the_offset_stays_inside_the_bounds_the_field_declares(engine):
    engine.traits["neuroticism"] = 99.0
    engine._baseline_traits["neuroticism"] = 0.0
    assert _mirror(engine)["neuroticism"] == pytest.approx(1.0)


def test_an_engine_that_cannot_be_reached_leaves_the_offsets_alone():
    growth = dict.fromkeys(_PERSONALITY_GROWTH_KEYS, 0.25)
    with mock.patch(
        "core.brain.personality_engine.get_personality_engine",
        side_effect=RuntimeError("no engine"),
    ):
        BondingPhase._mirror_the_drift(growth)
    assert growth == dict.fromkeys(_PERSONALITY_GROWTH_KEYS, 0.25)
