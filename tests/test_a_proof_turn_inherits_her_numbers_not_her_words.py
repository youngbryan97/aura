"""What survives the boundary between two turns of a proof run.

`clear_transient_response_modifiers(strict=True)` emptied the dict. Every
campaign the subject-core battery runs is a proof run, so on every turn of
every campaign her published measurements were thrown away before the first
phase read them: developmental novelty, which the body's novelty sense reads;
phi and the three gates it sets; the workspace's ignition level; the two
voices' agreement. The only value that crossed a turn boundary was whatever
snapshot a background ALife task happened to be holding, which is why one of
them read exactly 0.5 for three hundred rounds.

The rule the scrub now keeps is the one the isolation was for: a proof turn
must not inherit a word from the turn before it. A finite number is not a word.
"""

from __future__ import annotations

import math

import pytest

from core.runtime.proof_policy import (
    TRANSIENT_RESPONSE_MODIFIER_KEYS,
    _is_a_reading,
    clear_transient_response_modifiers,
)


# ── what counts as a reading ─────────────────────────────────────────────


@pytest.mark.parametrize(
    "value",
    [0.0, 1, -3.25, True, False, {"edges": 4.0}, [0.1, 0.2], (1, 2), {"a": {"b": 1.0}}],
)
def test_a_number_and_a_shape_of_numbers_are_readings(value):
    assert _is_a_reading(value) is True


@pytest.mark.parametrize(
    "value",
    [
        "observe_stabilize_replay",
        b"bytes",
        None,
        {},
        [],
        {"claim": "I am here"},
        ["open thread"],
        object(),
        float("nan"),
        math.inf,
    ],
)
def test_words_absences_and_objects_are_not_readings(value):
    assert _is_a_reading(value) is False


def test_nesting_stops():
    """Three levels down is carrying a structure, not a measurement."""
    assert _is_a_reading({"a": {"b": 1.0}}) is True
    assert _is_a_reading({"a": {"b": {"c": 1.0}}}) is False


# ── the scrub ────────────────────────────────────────────────────────────


def test_a_strict_turn_keeps_the_measurements():
    modifiers = {
        "ontogenetic_novelty": 0.2239,
        "phi": 0.565,
        "ignited": True,
        "phi_autonomy_scale": 1.04,
        "bicameral_dissent": 0.31,
        "topology_metrics": {"edges": 12.0, "nodes": 7.0},
    }
    kept = dict(modifiers)
    clear_transient_response_modifiers(modifiers, strict=True)
    assert modifiers == kept


def test_a_strict_turn_drops_every_word():
    modifiers = {
        "unity_claim": "one bound moment",
        "mood_narrative": "steady",
        "precomputed_grounded_reply": "here is the answer",
        "phi": 0.5,
    }
    clear_transient_response_modifiers(modifiers, strict=True)
    assert modifiers == {"phi": 0.5}


def test_a_strict_turn_drops_a_named_transient_even_when_it_is_a_number():
    """The named list runs first, so a numeric directive does not sneak through."""
    assert "deep_handoff" in TRANSIENT_RESPONSE_MODIFIER_KEYS
    modifiers = {"deep_handoff": True, "model_tier": 2, "phi": 0.5}
    clear_transient_response_modifiers(modifiers, strict=True)
    assert modifiers == {"phi": 0.5}


def test_an_ordinary_turn_is_unchanged_by_this():
    """Only the strict path moved. A live turn clears the named list and no more."""
    modifiers = {"conversational_dynamics": "stale", "unity_claim": "words", "phi": 0.5}
    clear_transient_response_modifiers(modifiers)
    assert modifiers == {"unity_claim": "words", "phi": 0.5}


def test_the_scrub_leaves_a_non_dict_alone():
    clear_transient_response_modifiers(None, strict=True)
    clear_transient_response_modifiers("not a dict", strict=True)


# ── the boundary itself ──────────────────────────────────────────────────


def test_the_turn_door_carries_the_novelty_into_the_next_turn(monkeypatch):
    """The body's novelty sense reads this before any phase writes it."""
    monkeypatch.setenv("AURA_PROOF_RUN", "1")
    from core.kernel.turn_door import clear_last_turn
    from core.state.aura_state import AuraState

    state = AuraState.default()
    state.response_modifiers["ontogenetic_novelty"] = 0.2239
    state.response_modifiers["precomputed_grounded_reply"] = "stale answer"
    clear_last_turn(state, "a new objective", "test")
    assert state.response_modifiers["ontogenetic_novelty"] == 0.2239
    assert "precomputed_grounded_reply" not in state.response_modifiers
