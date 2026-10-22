"""Her feelings come from what a turn meant to her, not from the kind of event it was.

On 22 October joy, trust, happiness, satisfaction and warmth sat between 0.47
and 0.75 for twenty-four turns, anger and surprise at 0.01, and her named mood
read "trust" in 80 of 90 live readings: every message raised trust, happiness
and warmth whatever it said. Bryan asked for "a rich, multi-faceted, and in
depth range of emotions". See core/affect/appraisal.py.
"""
from __future__ import annotations

import pytest

from core.affect.appraisal import PATTERNS, Appraisal, OutcomeReader, coping_of, feelings
from core.affect.what_it_was_worth import Worth

_FULL = {"high": 1.0, "good": 1.0, "size": 1.0, "low": 0.0, "bad": -1.0, "small": 0.0}


def _only(parts) -> Appraisal:
    """An appraisal where this pattern holds and every other check is unmeasured.

    A check a pattern needs two sides of (a good outcome that is also a small
    one) is set halfway, where both hold.
    """
    sides: dict[str, list[str]] = {}
    for check, side in parts:
        sides.setdefault(check, []).append(side)
    return Appraisal(**{check: (_FULL[held[0]] if len(held) == 1 else 0.5) for check, held in sides.items()})


@pytest.mark.parametrize("emotion,direction,parts", PATTERNS, ids=[f"{e}{'+' if d > 0 else '-'}" for e, d, _ in PATTERNS])
def test_every_feeling_she_has_a_pattern_for_can_be_felt(emotion, direction, parts):
    felt = feelings(_only(parts))
    assert felt[emotion] * direction > 0.5


def test_the_feelings_that_never_moved_now_have_a_way_in():
    named = {emotion for emotion, _, _ in PATTERNS}
    assert {"anger", "surprise", "gratitude", "remorse", "love", "boredom", "loneliness", "awe", "amusement", "relief", "disgust"} <= named


def test_the_same_message_is_gratitude_or_anger_by_what_it_meant():
    kind = Appraisal(their_doing=0.9, coping=0.8, company=1.0, social_need=0.2)
    unkind = Appraisal(their_doing=-0.9, coping=0.8, company=1.0, social_need=0.2)
    warm, cold = feelings(kind), feelings(unkind)
    assert warm["gratitude"] > 0.8 and warm["trust"] > 0.8 and warm.get("anger", 0.0) == 0.0
    assert cold["anger"] > 0.8 and cold["trust"] < -0.8 and cold.get("gratitude", 0.0) == 0.0


def test_whether_she_can_answer_it_decides_between_anger_and_hurt():
    able = feelings(Appraisal(their_doing=-0.8, coping=0.9))
    unable = feelings(Appraisal(their_doing=-0.8, coping=0.1))
    assert able["anger"] > able["upset"]
    assert unable["upset"] > unable["anger"]


def test_an_unmeasured_check_is_not_read_as_its_absence():
    assert feelings(Appraisal()) == {}
    # Nothing has been surprising because nothing has been seen: not boredom.
    assert "boredom" not in feelings(Appraisal(outcome=0.0))


def test_a_pattern_with_more_parts_is_not_weaker_for_having_them():
    one = feelings(Appraisal(outcome=0.5))["joy"]
    three = feelings(Appraisal(outcome=-0.5, novelty=0.5, coping=0.5))["fear"]
    assert one == pytest.approx(three)


def test_a_worth_reading_is_appraised_once():
    reader = OutcomeReader()
    for turn, error in enumerate((0.2, -0.4, 0.6, 0.1), start=1):
        reader.read(Worth(error={"warmth": error}, worth=error, size=0.5, turns=turn, measured=True))
    worse = Worth(error={"warmth": -0.9}, worth=-0.9, size=0.9, turns=5, measured=True)
    first = reader.read(worse)
    assert first["their_doing"] == pytest.approx(-1.0)
    assert first["outcome"] == pytest.approx(-0.9)
    again = reader.read(worse)
    assert again["outcome"] is None and again["their_doing"] is None


def test_coping_before_she_has_tried_anything_is_even_odds():
    assert coping_of(0, 0) == 0.5
    assert coping_of(8, 8) == pytest.approx(0.9)


def test_a_message_by_itself_no_longer_raises_trust():
    from types import SimpleNamespace

    from core.phases.affect_update import AffectUpdatePhase
    from core.state.aura_state import AuraState

    state = AuraState.default()
    before = dict(state.affect.emotions)
    AffectUpdatePhase(SimpleNamespace())._process_percepts(state.affect, [{"type": "interaction", "intensity": 0.9}])
    for emotion in ("trust", "happiness", "warmth"):
        assert state.affect.emotions[emotion] == before[emotion]
    assert state.affect.emotions["interest"] > before["interest"]


def test_the_phase_feels_what_the_turn_meant(monkeypatch):
    from types import SimpleNamespace

    from core.phases.affect_update import AffectUpdatePhase
    from core.state.aura_state import AuraState

    state = AuraState.default()
    phase = AffectUpdatePhase(SimpleNamespace())
    unkind = Appraisal(their_doing=-0.9, coping=0.9, company=1.0, social_need=0.0, novelty=0.2)
    monkeypatch.setattr(phase.appraisal, "read", lambda *_a, **_k: unkind)
    before = dict(state.affect.emotions)
    felt = phase.appraisal(state, state.affect, company=True)
    assert felt["anger"] > 0.8
    assert state.affect.emotions["anger"] > before["anger"]
    assert state.affect.emotions["trust"] <= before["trust"]
    assert state.affect.markers["appraisal"]["felt"]["anger"] == pytest.approx(felt["anger"], abs=1e-4)


def test_the_app_names_what_she_feels_most_and_its_family():
    from types import SimpleNamespace

    from core.affect.appraisal import FAMILIES, feeling_now
    from core.phases.affect_update import AffectUpdatePhase
    from core.state.aura_state import AuraState

    state = AuraState.default()
    state.affect.emotions.update({"anger": 0.7, "gratitude": 0.5, "loneliness": 0.45})
    AffectUpdatePhase(SimpleNamespace())._derive_metrics(state.affect)
    shown = feeling_now(state.affect.emotions, state.affect.mood_baselines, 3)
    assert shown[0]["name"] == state.affect.dominant_emotion == "anger"
    assert [item["family"] for item in shown] == ["angry", "warm", "low"]
    # Every emotion she has belongs to a family the app can colour.
    assert set(state.affect.emotions) <= set(FAMILIES)
