"""A narrative rendered from a state is not evidence about that state.

Give a language model a state and ask it to write, and what comes back carries
no information the state did not already have. Read it back as an observation
and the loop closes: confidence rises with every pass while nothing new has
been measured. Four of those were running.

Each test below fails if one returns.
"""

from __future__ import annotations

import time

import pytest

from core.consciousness.narrative_provenance import (
    FRESH_FOR_S,
    Grade,
    Rendering,
    RenderingLog,
    digest,
    dominant_label,
    evidence_grade,
    usable_as_evidence,
)


def test_a_rendering_is_not_evidence_for_its_own_state():
    log = RenderingLog()
    state = {"valence": 0.4, "arousal": 0.7}
    rendering = log.record("Everything feels sharp right now.", state, "narrative")
    assert evidence_grade(rendering, digest(state)) is Grade.SELF
    assert usable_as_evidence(rendering, digest(state)) is False


def test_a_rendering_of_an_earlier_state_is_evidence_about_that_state():
    log = RenderingLog()
    rendering = log.record("It was quiet then.", {"valence": 0.0}, "narrative")
    assert usable_as_evidence(rendering, digest({"valence": 0.6})) is True


def test_an_unattributed_narrative_is_treated_as_self_evidence():
    """The dangerous case is text with no state recorded beside it."""
    loose = Rendering(text="I feel steady", state_digest="", generator="?")
    assert evidence_grade(loose, digest({"x": 1.0})) is Grade.UNATTRIBUTED
    assert usable_as_evidence(loose, digest({"x": 1.0})) is False


def test_a_stale_rendering_is_not_evidence_for_now():
    old = Rendering(
        text="steady", state_digest="abc", generator="g",
        rendered_at=time.time() - FRESH_FOR_S - 1.0,
    )
    assert evidence_grade(old, "zzz") is Grade.STALE
    assert usable_as_evidence(old, "zzz") is False


def test_the_digest_ignores_text_derived_from_the_state():
    """Admitting strings would let a rendering change its own source digest."""
    base = {"valence": 0.5, "arousal": 0.2}
    with_text = dict(base, narrative="a long first-person sentence")
    assert digest(base) == digest(with_text)


def test_the_digest_survives_float_noise_but_not_a_real_change():
    assert digest({"v": 0.500001}) == digest({"v": 0.5000012})
    assert digest({"v": 0.5}) != digest({"v": 0.6})


def test_renderings_over_distinct_states_drop_repeats_of_one_state():
    """Counting across several renderings of one state counts the generator."""
    log = RenderingLog()
    for _ in range(4):
        log.record("still ponds again", {"v": 0.3}, "narrative")
    log.record("something shifted", {"v": 0.9}, "narrative")
    assert len(log.recent(5)) == 5
    assert len(log.over_distinct_states(5)) == 2


def test_dominant_label_does_not_depend_on_hash_order():
    """`max` over a dict built from a set breaks ties by insertion order."""
    assert dominant_label(["cognitive", "social", "cognitive", "social"]) == "cognitive"
    assert dominant_label(["social", "cognitive", "social", "cognitive"]) == "cognitive"
    assert dominant_label(["b", "b", "a"]) == "b"
    assert dominant_label([], default="neutral") == "neutral"


def test_synthesis_depth_no_longer_reads_the_narrative_it_produced():
    """A verbose model used to produce a richer moment."""
    from core.consciousness.stream_of_being import ExperienceIntegrator

    integrator = ExperienceIntegrator()
    moment = integrator.synthesize()
    before = moment.synthesis_depth
    moment.interior_text = "x" * 5000
    assert integrator._compute_synthesis_depth(moment) == pytest.approx(before), (
        "the length of the text generated FROM this moment changed a measure "
        "of how rich the moment is"
    )


def test_the_stream_narrative_state_excludes_its_own_text():
    from core.consciousness.stream_of_being import ExperienceIntegrator, _moment_state

    moment = ExperienceIntegrator().synthesize()
    first = digest(_moment_state(moment))
    moment.interior_text = "an entirely different narrative, much longer than before"
    assert digest(_moment_state(moment)) == first


def test_the_witness_is_not_shown_renderings_of_the_state_it_is_judging():
    import inspect

    from core.consciousness.phenomenological_experiencer import PhenomenalSelfModel

    source = inspect.getsource(PhenomenalSelfModel.run_witness_reflection)
    assert "usable_as_evidence" in source, (
        "the witness reads narratives again without checking what state each "
        "was rendered from"
    )
    assert "_phenomenal_reports" not in source, (
        "the witness is shown its own previous outputs as the experiential "
        "stream it is asked to find patterns in"
    )


def test_the_deep_narrative_is_not_written_from_the_last_one_of_the_same_state():
    import inspect

    from core.consciousness.stream_of_being import StreamOfBeing

    source = inspect.getsource(StreamOfBeing._run_deep_narrative)
    assert "usable_as_evidence" in source
    assert "m.interior_text" not in source, (
        "the prompt carries the previous interior text unconditionally again"
    )


def test_a_stale_rendering_does_not_enter_every_llm_call():
    import inspect

    from core.consciousness.phenomenological_experiencer import PhenomenalSelfModel

    source = inspect.getsource(PhenomenalSelfModel.get_phenomenal_context_fragment)
    assert "FRESH_FOR_S" in source, (
        "the latest introspection is injected with no freshness check, so a "
        "state nobody is in keeps being asserted"
    )
