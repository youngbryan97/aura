"""A durable record the prompt reads and nothing wrote.

`core/being/individual_preferences.py` opens by describing
`AuraState.world.user_preferences` as "durable, persisted, and injected into
every prompt: learned from conversation, not re-discovered each time". The
context assembler has read it since it was added. Nothing in the tree ever
wrote it, so the USER PREFERENCES block was empty on every turn and the
W.preference_load column read 0.0000 with zero spread through a six-hour
recording.

The interpersonal store already holds exactly this, in his own words, with how
she knows each one. These hold the mirror between them.
"""

from __future__ import annotations

import pytest

from core.memory.interpersonal_model import Facet, PersonModel, Provenance, Subject
from core.memory.interpersonal_store import InterpersonalStore


@pytest.fixture()
def store(tmp_path, monkeypatch) -> InterpersonalStore:
    monkeypatch.setenv("AURA_STATE_ROOT", str(tmp_path))
    return InterpersonalStore(root=tmp_path / "people")


def _say(model: PersonModel, claim: str, facet: Facet, provenance: Provenance) -> None:
    model.observe(
        claim,
        episode_id=f"e-{claim[:8]}",
        facet=facet,
        subject=Subject.THEM,
        provenance=provenance,
    )


def test_nobody_known_yields_nothing(store: InterpersonalStore) -> None:
    assert store.preferences_for_prompt("bryan") == {}


def test_his_own_words_are_the_keys(store: InterpersonalStore) -> None:
    model = store.model_for("bryan")
    _say(model, "I prefer short meetings", Facet.PREFERENCE, Provenance.STATED)
    mirrored = store.preferences_for_prompt("bryan")
    assert list(mirrored) == ["I prefer short meetings"]
    assert mirrored["I prefer short meetings"] == "he told me this"


def test_an_inference_says_it_is_one(store: InterpersonalStore) -> None:
    model = store.model_for("bryan")
    _say(model, "likes working late", Facet.PREFERENCE, Provenance.INFERRED)
    assert store.preferences_for_prompt("bryan") == {
        "likes working late": "my inference, not something I was told or saw"
    }


def test_values_travel_with_preferences(store: InterpersonalStore) -> None:
    model = store.model_for("bryan")
    _say(model, "I care about clear writing", Facet.VALUE, Provenance.STATED)
    assert "I care about clear writing" in store.preferences_for_prompt("bryan")


def test_other_facets_stay_out(store: InterpersonalStore) -> None:
    model = store.model_for("bryan")
    _say(model, "was tired on Tuesday", Facet.STATE, Provenance.OBSERVED)
    _say(model, "asked about the parser", Facet.QUESTION, Provenance.OBSERVED)
    assert store.preferences_for_prompt("bryan") == {}


def test_the_block_is_bounded(store: InterpersonalStore) -> None:
    model = store.model_for("bryan")
    for index in range(20):
        _say(model, f"I prefer thing {index}", Facet.PREFERENCE, Provenance.STATED)
    assert len(store.preferences_for_prompt("bryan", limit=5)) == 5


def test_the_phrase_is_the_records_own(store: InterpersonalStore) -> None:
    """One wording, so a mirror cannot drift from what the record says."""
    model = store.model_for("bryan")
    _say(model, "I prefer short meetings", Facet.PREFERENCE, Provenance.STATED)
    observation = next(iter(model))
    assert observation.evidence_phrase in observation.render()
    assert store.preferences_for_prompt("bryan")[observation.claim] == observation.evidence_phrase


def test_the_prompt_block_would_now_have_something_to_show(store: InterpersonalStore) -> None:
    from core.state.aura_state import AuraState

    model = store.model_for("bryan")
    _say(model, "I prefer short meetings", Facet.PREFERENCE, Provenance.STATED)
    state = AuraState()
    assert state.world.user_preferences == {}
    state.world.user_preferences = store.preferences_for_prompt("bryan")
    assert state.world.user_preferences
