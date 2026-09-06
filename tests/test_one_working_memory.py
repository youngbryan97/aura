"""One list, one capacity, and every reader normalising against it.

The defects this pins were all live:

* The sandbox's memory-bomb guard read ``state.working_memory``. The list is
  on ``state.cognition``, so the attribute was absent, the guard's condition
  was never evaluated, and a 5,000-item state passed structural validation.
* The context assembler passed ``working_memory_cap=40`` against an enforced
  capacity of 150, so the finitude model's ``context_usage`` read 1.0 from
  the fortieth exchange onward — into a block that reports Aura's own
  situation to Aura.
* The executive closure state vector divided by 20, pinning that element at
  its ceiling for almost every conversation.
"""
from __future__ import annotations

import pytest

from core.state.aura_state import AuraState, MAX_WORKING_MEMORY
from core.state.one_working_memory import (
    THE_STORE,
    how_full,
    the_capacity,
    the_caps_that_disagree,
    the_working_memory,
    who_else_holds_it,
)


def test_the_capacity_is_the_one_the_trimmer_enforces():
    """Not a second number that happens to agree today."""
    assert the_capacity() == MAX_WORKING_MEMORY


def test_the_accessor_finds_the_list_on_either_shape():
    """The guard that broke took an AuraState and asked it for the list."""
    state = AuraState()
    state.cognition.working_memory = [{"role": "user", "content": "hello"}]

    assert the_working_memory(state) == state.cognition.working_memory
    assert the_working_memory(state.cognition) == state.cognition.working_memory


def test_the_accessor_returns_a_list_for_something_that_has_neither():
    assert the_working_memory(object()) == []
    assert the_working_memory(None) == []


def test_a_list_that_is_not_a_list_reads_as_empty_rather_than_raising():
    class Odd:
        working_memory = "not a list"

    assert the_working_memory(Odd()) == []


def test_how_full_is_not_clamped_at_the_ceiling():
    """Over-full is a real condition and the one worth seeing.

    The trimmer runs after the append, so a reader legitimately sees one more
    than the capacity. Reporting that as exactly full is the same information
    loss as the caps this module exists to remove.
    """
    state = AuraState()
    state.cognition.working_memory = [{"i": i} for i in range(the_capacity() + 1)]
    assert how_full(state) > 1.0


def test_how_full_is_zero_on_an_empty_mind():
    assert how_full(AuraState()) == 0.0


def test_nothing_normalises_the_working_memory_against_its_own_number():
    """The gate. Baseline zero, and it only goes down."""
    disagreeing = the_caps_that_disagree()
    assert disagreeing == [], (
        "these sites divide or cap the working memory by a number that is not "
        f"the enforced capacity ({the_capacity()}); ask the_capacity() "
        f"instead: {disagreeing}"
    )


def test_every_other_holder_names_its_derivation_and_freshness():
    """A holder that cannot say what it derives from is a fork, not a view."""
    for holder in who_else_holds_it():
        assert holder["who"]
        assert holder["holds"]
        assert holder["derived"], f"{holder['who']} does not say what it derives from"
        assert holder["fresh"], f"{holder['who']} does not say how fresh it is"


def test_the_store_is_addressed_by_one_name():
    assert THE_STORE == "cognition.working_memory"
    state = AuraState()
    holder: object = state
    for part in THE_STORE.split("."):
        holder = getattr(holder, part)
    assert isinstance(holder, list)


# ----------------------------------------------------- the guard that broke


def test_the_sandbox_refuses_a_memory_bomb():
    from core.kernel.shadow_kernel import ShadowExecutionPhase

    state = AuraState()
    state.cognition.working_memory = [{"role": "user", "content": "x" * 200}
                                      for _ in range(5000)]
    phase = object.__new__(ShadowExecutionPhase)
    assert ShadowExecutionPhase._validate_state_bounds(phase, state) is False


def test_the_sandbox_allows_a_state_one_item_past_the_trim():
    """The append happens before the trim; that state is not an attack."""
    from core.kernel.shadow_kernel import ShadowExecutionPhase

    state = AuraState()
    state.cognition.working_memory = [{"i": i} for i in range(the_capacity() + 1)]
    phase = object.__new__(ShadowExecutionPhase)
    assert ShadowExecutionPhase._validate_state_bounds(phase, state) is True


def test_the_bomb_threshold_is_derived_and_not_written_down():
    from core.kernel.shadow_kernel import StateBoundsConfig

    config = StateBoundsConfig()
    assert not hasattr(config, "MAX_WORKING_MEMORY_ITEMS")
    assert config.WORKING_MEMORY_HEADROOM >= 1


# ------------------------------------------------- the readers that saturated


def test_the_closure_vector_still_moves_past_twenty_exchanges():
    from core.consciousness.executive_closure import (
        _how_full_the_working_memory_is,
    )

    state = AuraState()
    state.cognition.working_memory = [{"i": i} for i in range(21)]
    at_twenty_one = _how_full_the_working_memory_is(state)
    state.cognition.working_memory = [{"i": i} for i in range(100)]
    at_a_hundred = _how_full_the_working_memory_is(state)

    assert at_twenty_one < at_a_hundred < 1.0


def test_the_finitude_model_is_asked_for_the_real_capacity():
    """The assembler's call site, read rather than executed.

    Executing it needs a whole assembled context; the defect was a literal in
    the argument, which is visible in the source and cannot be misread.
    """
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1]
        / "core" / "brain" / "llm" / "context_assembler.py"
    ).read_text("utf-8")
    assert "working_memory_cap=40" not in source
    assert "working_memory_cap=the_capacity()" in source


@pytest.mark.parametrize("size", [0, 1, 40, 149, 150, 151, 5000])
def test_occupancy_is_monotonic_in_the_number_of_exchanges(size):
    state = AuraState()
    state.cognition.working_memory = [{"i": i} for i in range(size)]
    assert how_full(state) == pytest.approx(size / the_capacity())
