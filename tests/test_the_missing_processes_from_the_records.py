"""Dignified need, joint recall and the wordless channel.

Three processes the organ table had counted as covered on the strength of a
docstring. These pin the mechanism each one has now.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.brain.response_quality import extract_features
from core.brain.taste_model import FEATURE_PRIORS
from core.expression.register import read
from core.phases.memory_retrieval import _shared_with
from core.social.telling import worth_telling
from core.state.aura_state import AuraState
from core.voice.duplex.prosody import _GAIN_CEILING, _PAUSE_CEILING_MS, ProsodySpec, carry_breakthrough

PLEA = "Can you please help me with this? Could you just do it for me?"
OFFER = (
    "Could you look at the schedule with me? If I send you the logs, I can walk you "
    "through what I found."
)


# ── dignified need ────────────────────────────────────────────────────────


def test_the_register_measures_what_a_request_offers() -> None:
    assert read(OFFER).offering > 0.0
    assert read(PLEA).offering == 0.0


def test_a_request_that_also_gives_scores_above_a_plea() -> None:
    offered = extract_features(OFFER, user_message="hello there")["dignity"]
    pleaded = extract_features(PLEA, user_message="hello there")["dignity"]
    assert offered > pleaded == 0.0


def test_an_offer_nobody_asked_for_is_not_scored() -> None:
    telling = "I can send you the logs later, and I will walk you through what I found."
    assert extract_features(telling, user_message="hello there")["dignity"] == 0.0


def test_dignity_carries_the_prior_of_the_other_shape_fits() -> None:
    assert FEATURE_PRIORS["dignity"] == pytest.approx(FEATURE_PRIORS["register_match"])


# ── joint recall ──────────────────────────────────────────────────────────


def test_a_record_written_with_the_person_she_is_with_is_shared() -> None:
    assert _shared_with({"principal_id": "Bryan"}, "bryan")
    assert _shared_with({"user_id": "bryan"}, "bryan")


def test_a_record_with_somebody_else_or_nobody_is_not() -> None:
    assert not _shared_with({"principal_id": "someone_else"}, "bryan")
    assert not _shared_with({}, "bryan")
    assert not _shared_with({"principal_id": "local_user"}, "local_user")


def test_a_shared_recollection_is_passed_on_as_theirs_together() -> None:
    state = AuraState.default()
    state.cognition.relived = {"relived": True, "intensity": 0.6, "shared": True}
    state.cognition.long_term_memory = ["the night the migration finally finished"]
    reading = worth_telling(state.affect, state.cognition)
    assert reading.kind == "something we both remember"
    assert "migration" in reading.about


def test_a_recollection_she_had_alone_is_still_hers() -> None:
    state = AuraState.default()
    state.cognition.relived = {"relived": True, "intensity": 0.6, "shared": False}
    state.cognition.long_term_memory = ["a note I wrote to myself"]
    assert worth_telling(state.affect, state.cognition).kind == "something I remembered"


# ── the wordless channel ──────────────────────────────────────────────────


def _spec() -> ProsodySpec:
    return ProsodySpec(voice="af_heart", speed=1.0, gain=0.98, trailing_pause_ms=60.0)


def test_no_breakthrough_leaves_the_voice_as_compiled() -> None:
    assert carry_breakthrough(_spec(), SimpleNamespace(breakthrough=False, delivery_z=4.0)) == _spec()
    assert carry_breakthrough(_spec(), None) == _spec()


def test_a_breakthrough_is_held_with_more_air_and_a_fuller_voice() -> None:
    carried = carry_breakthrough(_spec(), SimpleNamespace(breakthrough=True, delivery_z=3.0))
    assert carried.trailing_pause_ms > _spec().trailing_pause_ms
    assert carried.gain > _spec().gain
    assert carried.speed == _spec().speed, "the loudest moment is held, not hurried"


def test_a_larger_breakthrough_moves_it_further_and_never_past_the_ceilings() -> None:
    small = carry_breakthrough(_spec(), SimpleNamespace(breakthrough=True, delivery_z=1.5))
    large = carry_breakthrough(_spec(), SimpleNamespace(breakthrough=True, delivery_z=40.0))
    assert large.trailing_pause_ms > small.trailing_pause_ms
    assert large.trailing_pause_ms <= _PAUSE_CEILING_MS
    assert large.gain <= _GAIN_CEILING
