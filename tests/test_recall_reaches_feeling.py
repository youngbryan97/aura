"""A reader with no writer is a channel that cannot fire.

The affect phase has carried a mapping from a `memory_replay` percept to
sadness, joy, trust, nostalgia, warmth and belonging since it was written, and
nothing in the tree ever emitted that percept. So recall could put something in
front of her and her feeling never heard about it — active memory reached
attention and nothing else, which is a spread of two domains in nine.
"""

from __future__ import annotations

from core.phases.affect_update import AffectUpdatePhase
from core.state.aura_state import AuraState
from core.state.percepts import emit_percept


def test_the_affect_phase_still_knows_what_a_replayed_memory_feels_like() -> None:
    """The mapping the emitter is written against."""
    import inspect

    source = inspect.getsource(AffectUpdatePhase._process_percepts)
    assert '"memory_replay"' in source, (
        "the percept kind recall emits is no longer in the emotion map"
    )


def test_something_coming_back_to_her_moves_how_she_feels() -> None:
    phase = AffectUpdatePhase(None)
    state = AuraState.default()
    affect = state.affect
    before = {name: float(value) for name, value in affect.emotions.items()}

    percept = emit_percept(
        state.world,
        "memory_replay",
        content="the afternoon we worked this out",
        intensity=0.8,
        source="memory_retrieval",
    )
    assert percept is not None
    phase._process_percepts(affect, [percept])

    moved = [
        name
        for name, value in affect.emotions.items()
        if abs(float(value) - before.get(name, 0.0)) > 1e-9
    ]
    assert moved, "a memory came back and nothing about her changed"


def test_a_faint_recollection_moves_her_faintly() -> None:
    """The intensity is the match score, so there is no threshold to choose."""
    phase = AffectUpdatePhase(None)

    def reach(intensity: float) -> float:
        state = AuraState.default()
        affect = state.affect
        before = {name: float(value) for name, value in affect.emotions.items()}
        percept = emit_percept(
            state.world, "memory_replay", content="x", intensity=intensity
        )
        phase._process_percepts(affect, [percept])
        return sum(
            abs(float(value) - before.get(name, 0.0))
            for name, value in affect.emotions.items()
        )

    assert reach(0.9) > reach(0.1)


def test_retrieval_emits_it() -> None:
    import inspect

    from core.phases import memory_retrieval

    source = inspect.getsource(memory_retrieval)
    assert 'emit_percept(' in source
    assert '"memory_replay"' in source
