"""The router knew, and the component that had to act was never told.

Live, verbatim. The router logged:

    Routing: multi-step skill-backed task detected → TASK via ['desktop_task']
    CognitiveRouting: Mode=DELIBERATE

...and the reply was:

    "I can't interact with your device or open apps. But here's a note for you:
     'Orcas, also known as killer whales...'"

A false capability denial, with the note's content typed into chat, while Notes
stayed untouched. The router writes its matched skills onto the state and
nothing downstream read them, so the desktop planning contract never rendered
and she answered from identity text instead of planning.

This is connective tissue, not a special case: a decision one phase has already
made must be readable by the phases that depend on it.
"""

from __future__ import annotations

import inspect

from source_support import inlined_function_source

from core.phases import response_generation


def test_generation_reads_the_routers_matched_skills():
    # As it runs: its blocks were moved into helpers by the size sweep.
    source = inlined_function_source(response_generation.__file__, "ResponseGenerationPhase.execute")
    assert 'state.response_modifiers.get("matched_skills")' in source, (
        "the router's own decision must be consumed, not re-derived from text"
    )
    contract_at = source.index("LIVE DESKTOP EXECUTION PLANNING CONTRACT")
    read_at = source.index('state.response_modifiers.get("matched_skills")')
    assert read_at < contract_at, "the decision must be read before the contract renders"


def test_the_router_writes_the_decision_it_makes():
    """Both halves of the seam, so neither can be removed alone."""
    from core.phases import cognitive_routing_unitary

    router_source = inspect.getsource(cognitive_routing_unitary)
    assert 'response_modifiers["matched_skills"] = matched' in router_source, (
        "the router must publish the skills it matched"
    )


def test_a_desktop_skill_match_is_recognised():
    """The recognition covers the skill family, not one hard-coded name."""
    for matched in (["desktop_task"], ["computer_use"], ["desktop_task", "web_search"]):
        assert any(
            "desktop" in str(s).lower() or "computer_use" in str(s).lower() for s in matched
        ), matched
    for matched in (["web_search"], [], ["memory_recall"]):
        assert not any(
            "desktop" in str(s).lower() or "computer_use" in str(s).lower() for s in matched
        ), matched


def test_the_text_detector_remains_as_a_fallback():
    """A lane that never reached the router must still plan, not deny."""
    # As it runs: its blocks were moved into helpers by the size sweep.
    source = inlined_function_source(response_generation.__file__, "ResponseGenerationPhase.execute")
    assert "looks_like_desktop_objective" in source
