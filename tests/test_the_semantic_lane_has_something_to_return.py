"""A queue with a producer and no consumer.

`MemoryConsolidationPhase` appended every committed piece of knowledge to the
evolution log with the comment "Queue the knowledge for the ColdStore to
process asynchronously". Nothing processed it. `cold.long_term_memory` had no
writer anywhere in the tree, and the subject-core driver registers exactly that
list as the SEMANTIC lane of the intentional retriever.

The consequence was measurable rather than theoretical: through a six-hour
campaign the M.cold_memory_load column read 0.0000 with a standard deviation of
zero, while retrieved_load and recall_score moved the whole time. Her semantic
memory returned nothing, every time, and the flat column said so to anyone who
looked.
"""

from __future__ import annotations

from core.phases.memory_consolidation import MemoryConsolidationPhase
from core.state.aura_state import MAX_SEMANTIC_MEMORY, AuraState

absorb = MemoryConsolidationPhase._absorb_semantic


def _state() -> AuraState:
    return AuraState()


def test_committed_knowledge_reaches_the_semantic_store() -> None:
    state = _state()
    assert state.cold.long_term_memory == []
    absorb(state, "User: where did we leave the parser\nAura: at the lexer")
    assert state.cold.long_term_memory == [
        "User: where did we leave the parser\nAura: at the lexer"
    ]


def test_an_exact_repeat_of_the_last_entry_is_not_written_twice() -> None:
    state = _state()
    absorb(state, "the same thing")
    absorb(state, "the same thing")
    assert state.cold.long_term_memory == ["the same thing"]


def test_the_same_thing_said_again_later_is_two_occasions() -> None:
    state = _state()
    absorb(state, "the same thing")
    absorb(state, "something else")
    absorb(state, "the same thing")
    assert state.cold.long_term_memory == [
        "the same thing",
        "something else",
        "the same thing",
    ]


def test_nothing_is_written_for_nothing() -> None:
    state = _state()
    absorb(state, "")
    absorb(state, "   ")
    absorb(state, None)  # type: ignore[arg-type]
    assert state.cold.long_term_memory == []


def test_the_store_stays_within_its_measured_bound() -> None:
    state = _state()
    for index in range(MAX_SEMANTIC_MEMORY + 5):
        absorb(state, f"memory {index}")
    assert len(state.cold.long_term_memory) == MAX_SEMANTIC_MEMORY
    # The oldest go, not the newest.
    assert state.cold.long_term_memory[-1] == f"memory {MAX_SEMANTIC_MEMORY + 4}"


def test_a_state_with_no_cold_store_is_left_alone() -> None:
    state = _state()
    state.cold = None
    absorb(state, "something")  # must not raise


def test_the_semantic_lane_now_has_a_writer() -> None:
    """The lane the subject-core driver registers reads this list."""
    from pathlib import Path

    driver = Path("core/subject/driver.py").read_text(encoding="utf-8")
    assert "runtime.state.cold.long_term_memory" in driver
    assert "MemoryStoreType.SEMANTIC" in driver

    consolidation = Path("core/phases/memory_consolidation.py").read_text(encoding="utf-8")
    assert "_absorb_semantic" in consolidation
