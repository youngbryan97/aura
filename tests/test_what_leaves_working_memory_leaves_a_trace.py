"""Two mechanisms at one bound, and the one that records never ran.

`AuraState.compact` says compaction is the only moment where the original
wording is still available, folds what is leaving into the continuity ledger,
and forms her own positions from it. It fires above MAX_WORKING_MEMORY.
`trim_working_memory` prunes down to MAX_WORKING_MEMORY and is called from
several phases on every turn, so it holds the list at exactly the threshold the
fold waits to be crossed.

The result was measurable: through the six-hour run_032
`cognition.continuity_ledger`, `cognition.rolling_summary` and
`identity.self_preferences` all read zero with zero spread, and every turn that
left working memory left no trace at all.

The ledger sees what leaves, whichever path removes it, and her positions are
rebuilt from the ledger rather than accumulated onto what is already stored —
because this now runs on every commit, and accumulating would register one
position fifty times.
"""

from __future__ import annotations

from core.being.individual_preferences import IndividualPreferences, formation_threshold
from core.state.aura_state import MAX_WORKING_MEMORY, AuraState


def _turn(index: int, role: str = "user", text: str = "") -> dict:
    return {"role": role, "content": text or f"we established point {index} today", "metadata": {}}


SAID = "I think I really like slow music."


def test_a_prune_folds_what_it_drops() -> None:
    state = AuraState()
    state.cognition.working_memory = [_turn(i) for i in range(MAX_WORKING_MEMORY + 40)]
    assert not state.cognition.continuity_ledger

    state.cognition.salience_prune(MAX_WORKING_MEMORY)

    assert len(state.cognition.working_memory) == MAX_WORKING_MEMORY
    assert state.cognition.continuity_ledger, "the dropped turns left no trace"
    assert state.cognition.continuity_ledger.get("entries")


def test_a_prune_that_drops_nothing_writes_nothing() -> None:
    state = AuraState()
    state.cognition.working_memory = [_turn(i) for i in range(10)]
    state.cognition.salience_prune(MAX_WORKING_MEMORY)
    assert not state.cognition.continuity_ledger


def test_her_positions_form_without_a_compaction() -> None:
    """The gate compaction waits on is the one the trimmer never lets open."""
    state = AuraState()
    said = "I think I really like slow music."
    state.cognition.working_memory = [
        _turn(i, role="assistant", text=said) for i in range(MAX_WORKING_MEMORY + 40)
    ]
    state.cognition.salience_prune(MAX_WORKING_MEMORY)

    assert state.compact() is False, "this test is about the path where it does not fire"
    formed = IndividualPreferences.from_dict(state.identity.self_preferences)
    assert formed.items, "nothing accumulated from her own positions"


def test_forming_twice_does_not_count_the_same_position_twice() -> None:
    state = AuraState()
    said = "I think I really like slow music."
    state.cognition.working_memory = [
        _turn(i, role="assistant", text=said) for i in range(MAX_WORKING_MEMORY + 40)
    ]
    state.cognition.salience_prune(MAX_WORKING_MEMORY)

    state.compact()
    once = IndividualPreferences.from_dict(state.identity.self_preferences)
    first = {key: item.encounters for key, item in once.items.items()}
    for _ in range(5):
        state.compact()
    again = IndividualPreferences.from_dict(state.identity.self_preferences)
    assert {key: item.encounters for key, item in again.items.items()} == first


def test_the_replay_is_bounded_by_where_strength_saturates() -> None:
    """A pinned entry can carry a large mention count; the work is capped."""
    state = AuraState()
    said = "I think I really like slow music."
    state.cognition.working_memory = [
        _turn(i, role="assistant", text=said) for i in range(MAX_WORKING_MEMORY + 60)
    ]
    state.cognition.salience_prune(MAX_WORKING_MEMORY)
    state.compact()
    formed = IndividualPreferences.from_dict(state.identity.self_preferences)
    for item in formed.items.values():
        assert item.encounters <= formation_threshold() * 3


def test_a_position_and_its_polarity_can_be_read_from_one_sentence() -> None:
    """Two anchored patterns that could never match the same sentence.

    The ledger marks a sentence as a position when it opens with "I think",
    "I believe", "in my view" and the rest. Every stance pattern is anchored to
    the start of the sentence too, so a sentence that opened with one could not
    open with the other: every position entry read as polarity-unknown, the
    former recorded nothing for it, and no preference of her own could form.
    """
    from core.brain.llm.continuity_ledger import classify
    from core.state.aura_state import _stance_of_position

    for text, stance in (
        ("I think I really like slow music.", "drawn_to"),
        ("My position is that I dislike brutalist architecture.", "averse_to"),
        ("In my view I prefer slow music.", "drawn_to"),
        ("I believe I don't enjoy long meetings.", "averse_to"),
    ):
        assert classify(text, "assistant") == "position", text
        assert _stance_of_position(text) == stance, text


def test_an_opinion_with_no_polarity_still_records_nothing() -> None:
    """Guessing in the ambiguous case is the old defect with a smaller blast."""
    from core.state.aura_state import _stance_of_position

    assert _stance_of_position("I think the parser is slow.") is None
