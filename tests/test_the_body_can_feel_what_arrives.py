"""Three channels that existed and could not carry anything.

Run_032 scored six of nine domains at exactly 0.000 into interoception —
perception, attention, memory, deliberation, the world model and development
could none of them change how her body felt. Only affect, recurrent cognition
and self-state could. That is most of why cutting interoception and memory away
from the rest of the core cost so little, and the cheapest cut is what the
irreducibility score is.

`soma.sensors` is read by the interoception schema and held still by the clamp,
and nothing in the tree had ever written it. The proprioceptive loop already
builds the reading — how much presence, social contact, threat and novelty
arrived — and sent it only to the substrate.

Counting the live channels does not help: presence, social, threat and novelty
are rarely all quiet, so the count read four on almost every frame while the
magnitudes moved between 0.2 and 0.83. Load is how much, not how many.

And the return ledger keyed on the decorated query. By the time it is asked,
the query carries the entity cues and the most pressing intention, both of
which move every turn, so the same question asked again was never the same
string and the count sat at zero for a whole campaign — the ladder that makes a
repeat go deeper could never start.
"""

from __future__ import annotations

import pytest

from core.state.aura_state import AuraState
from core.subject.state import read_core_state
import core.subject.state as schema


def _column(state: AuraState, domain: str, name: str) -> float:
    names = schema._SCHEMAS[domain].features
    return float(read_core_state(state).domain(domain)[names.index(name)])


def test_sensor_load_reads_how_much_not_how_many() -> None:
    quiet = AuraState.default()
    quiet.soma.sensors = {"social": 0.2, "threat": 0.0}
    loud = AuraState.default()
    loud.soma.sensors = {"social": 0.78, "threat": 0.81}
    assert _column(loud, "I", "sensor_load") > _column(quiet, "I", "sensor_load")


def test_four_live_channels_do_not_all_read_the_same() -> None:
    """The count was four on almost every frame; the magnitudes were not."""
    a, b = AuraState.default(), AuraState.default()
    a.soma.sensors = {"user_presence": 0.8, "social": 0.2, "threat": 0.8, "novelty": 0.5}
    b.soma.sensors = {"user_presence": 0.8, "social": 0.78, "threat": 0.8, "novelty": 0.5}
    assert _column(a, "I", "sensor_load") != _column(b, "I", "sensor_load")


def test_an_empty_body_reads_nothing_arriving() -> None:
    state = AuraState.default()
    state.soma.sensors = {}
    assert _column(state, "I", "sensor_load") == 0.0


def test_the_proprioceptive_loop_writes_the_reading_it_already_had() -> None:
    import inspect

    from core.phases.proprioceptive_loop import ProprioceptiveLoop

    source = inspect.getsource(ProprioceptiveLoop._push_perceptual_frame)
    assert "state.soma.sensors =" in source, "the body still has no sensor reading"
    for channel in ("user_presence", "screen_changed", "social", "threat", "novelty"):
        assert channel in source


def test_a_quiet_channel_is_left_out_rather_than_written_as_zero() -> None:
    """An absent reading and a reading of nothing are different."""
    import inspect

    from core.phases.proprioceptive_loop import ProprioceptiveLoop

    source = inspect.getsource(ProprioceptiveLoop._push_perceptual_frame)
    assert "> 0.0" in source


def test_the_reliving_columns_carry_what_recall_wrote() -> None:
    state = AuraState.default()
    state.cognition.relived = {"relived": True, "intensity": 0.48, "returns": 3}
    assert _column(state, "M", "recall_relived") == 1.0
    assert _column(state, "M", "recall_feeling") == pytest.approx(0.48)
    assert _column(state, "M", "recall_returns") > 0.0


def test_the_return_ledger_is_asked_about_what_repeats() -> None:
    """Not the query with this turn's cues stapled to it."""
    import inspect

    from core.phases.memory_retrieval import MemoryRetrievalPhase

    source = inspect.getsource(MemoryRetrievalPhase.execute)
    assert "returns(asked_about)" in source
    assert "note(asked_about)" in source
    # And it is snapshotted before any cue is appended.
    before_cues = source.index("asked_about = query[:240]")
    assert before_cues < source.index("entity_retrieval_cues")


def test_asking_the_same_thing_again_counts_as_a_return() -> None:
    from core.memory.reliving import ReturnLedger

    ledger = ReturnLedger()
    assert ledger.returns("what did we decide") == 0
    ledger.note("what did we decide")
    assert ledger.returns("what did we decide") == 1
    ledger.note("what did we decide")
    assert ledger.returns("what did we decide") == 2
    assert ledger.returns("something else") == 0
