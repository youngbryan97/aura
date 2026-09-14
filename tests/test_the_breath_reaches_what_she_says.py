"""The delivery reading reaches the reply, the subject core, and the lesion.

`core/expression/delivery.py` reads the level she speaks from, what breaks
through it, and how long a breath is this cycle. A reading nothing consumes is
a mechanism that cannot fire, so these pin the three places it has to arrive:
the affect phase writes it, the reply is sized to the breath, and the clamp
that holds A still holds it too.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.expression import delivery
from core.phases.affect_update import AffectUpdatePhase
from core.phases.response_generation_unitary import _breath_in_words
from core.state.aura_state import AuraState
from core.subject.clamp import CLAMPED_FIELDS
from core.subject.state import _SCHEMAS


@pytest.fixture(autouse=True)
def _fresh_ledger():
    delivery.reset_for_test()
    yield
    delivery.reset_for_test()


def test_the_affect_phase_writes_the_reading_where_both_readers_look() -> None:
    state = AuraState.default()
    state.affect.valence = 0.4
    AffectUpdatePhase(SimpleNamespace())._read_delivery(state, state.affect)
    row = state.response_modifiers.get("delivery")
    assert row, "the reply path has nothing to size itself by"
    assert row["phrase_budget"] > 0
    assert state.affect.steadiness == pytest.approx(row["steadiness"])
    assert state.affect.lift == pytest.approx(row["lift"])
    assert state.affect.breakthrough is row["breakthrough"]


def test_a_breakthrough_reaches_affect_once_her_level_is_known() -> None:
    phase = AffectUpdatePhase(SimpleNamespace())
    state = AuraState.default()
    for level in (0.20, 0.24, 0.18, 0.22, 0.21, 0.19):
        state.affect.valence = level
        phase._read_delivery(state, state.affect)
    state.affect.valence = 0.95
    phase._read_delivery(state, state.affect)
    assert state.affect.breakthrough is True
    assert state.affect.delivery_z > 1.0


def test_the_breath_is_converted_with_this_drafts_own_word_length() -> None:
    state = AuraState.default()
    state.response_modifiers["delivery"] = {"phrase_budget": 400}
    short_words = "I am so glad you came by to see me."
    long_words = "Considerable uncertainty notwithstanding, expectations materialised."
    assert _breath_in_words(state, short_words) > _breath_in_words(state, long_words)
    per_word = len(short_words) / 10
    assert _breath_in_words(state, short_words) == round(400 / per_word)


def test_no_reading_leaves_the_taste_models_own_target_in_charge() -> None:
    state = AuraState.default()
    assert _breath_in_words(state, "anything at all") == 0
    state.response_modifiers["delivery"] = {"phrase_budget": 400}
    assert _breath_in_words(state, "") == 0


def test_the_columns_that_read_delivery_are_held_by_the_clamp() -> None:
    sources = set(_SCHEMAS["A"].sources)
    for field in ("affect.delivery_z", "affect.breakthrough", "affect.steadiness", "affect.lift"):
        assert field in sources, field
        assert field in CLAMPED_FIELDS["A"], field


def test_confirmation_is_one_column_not_two() -> None:
    """A duplicated column adds a coordinate with no dynamics of its own."""
    assert list(_SCHEMAS["A"].sources).count("affect.confirmation") == 1
