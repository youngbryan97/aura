"""The biggest live actuator, made answerable to the apparatus that measures.

The affective circumplex reaches user-facing generation as NUMBERS — sampling
temperature and the token budget, read by ``core/brain/inference_gate.py`` on
every non-background, non-isolated turn. Driving it across its range moves
temperature 0.500 -> 0.858 and the budget 472 -> 768. That is the largest
direct actuation in the system.

It was also the only one with no registered lesion. ``core/verify`` could ask
"does this faculty change the output?" of steering alpha, recurrent loops,
spiking bias, imagination bias and bicameral bias — and could not ask it of
the faculty with the visibly largest swing, because no counterfactual existed.

Note the near-miss this closes: ``affect.generation_controls`` WAS declared and
IS lesionable, but it belongs to ``core/being/affective_valence.py`` — a
different engine, on a different path. Two affect systems, and the measurable
one was not the live one. Distinct ids on purpose; reusing one would have made
two different measurements look like a single channel across boots.
"""

from __future__ import annotations

import pytest

from core.verify import influence_channels
from core.verify.lesion_registry import apply_channel, get_lesion_registry
from tests.source_contract import family_text

CHANNEL = influence_channels.AFFECT_CIRCUMPLEX_SAMPLING


@pytest.fixture(autouse=True)
def _circumplex_imported():
    """Registration happens at import, and the gate imports lazily per turn."""
    import core.affect.affective_circumplex  # noqa: F401


def test_the_live_affect_channel_has_a_lesion():
    assert get_lesion_registry().is_registered(CHANNEL), (
        "the circumplex reaches the sampler and cannot be measured"
    )


def test_it_is_declared_as_direct_actuation():
    """Numbers into the sampler, not a sentence in a prompt."""
    assert CHANNEL in influence_channels.DIRECT_ACTUATION_CHANNELS
    handle = get_lesion_registry().get(CHANNEL)
    assert handle is not None and handle.direct_actuation is True


def test_it_is_a_distinct_id_from_the_being_engine():
    """Two affect systems. One id each, or the ledger conflates them."""
    assert CHANNEL != influence_channels.AFFECT_GENERATION_CONTROLS


def test_lesioning_neutralises_the_temperature_and_restores_it():
    from core.affect.affective_circumplex import get_circumplex

    params = get_circumplex().get_llm_params()
    intact = apply_channel(CHANNEL, params["temperature"], neutral=None)
    assert isinstance(intact, float)

    with get_lesion_registry().lesion(CHANNEL):
        assert apply_channel(CHANNEL, params["temperature"], neutral=None) is None

    assert apply_channel(CHANNEL, params["temperature"], neutral=None) == intact


def test_none_is_the_correct_neutral_for_the_gate():
    """The gate initialises somatic_temperature to None and handles it.

    Lesioning must produce the no-affect path the gate already has, not a
    novel state — otherwise the counterfactual arm measures a code path that
    never runs in production.
    """

    from core.brain import inference_gate

    source = family_text(inference_gate)
    assert "somatic_temperature: float | None = None" in source
    assert "somatic_temperature if somatic_temperature is not None else 0.72" in source


def test_the_circumplex_actually_moves_the_numbers():
    """A lesion on an inert channel would prove nothing. This one is not inert."""
    from core.affect.affective_circumplex import get_circumplex

    circumplex = get_circumplex()
    seen_temp: set[float] = set()
    seen_tokens: set[int] = set()
    for valence_delta, arousal_delta in ((0.9, 0.9), (-1.8, 0.0), (0.0, -1.5)):
        circumplex.apply_event(valence_delta, arousal_delta)
        params = circumplex.get_llm_params()
        seen_temp.add(round(float(params["temperature"]), 3))
        seen_tokens.add(int(params["max_tokens"]))

    assert len(seen_temp) > 1, f"temperature never moved: {seen_temp}"
    assert len(seen_tokens) > 1, f"token budget never moved: {seen_tokens}"
    assert max(seen_temp) - min(seen_temp) > 0.1, (
        f"temperature swing too small to be worth measuring: {seen_temp}"
    )
