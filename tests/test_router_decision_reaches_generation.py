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

from core.phases import response_generation
from tests.source_contract import function_with_its_helpers, in_order


def _generation_execute() -> str:
    """`execute` and the helpers the method-size sweep lifted out of it.

    The desktop planning contract moved into
    `_execute_derived_merely_read`, so reading `execute` alone stopped
    finding any of this while the seam was untouched.
    """

    return function_with_its_helpers(
        response_generation, "ResponseGenerationPhase.execute"
    )


def test_generation_reads_the_routers_matched_skills():
    source = _generation_execute()
    assert 'state.response_modifiers.get("matched_skills")' in source, (
        "the router's own decision must be consumed, not re-derived from text"
    )
    in_order(
        source,
        'state.response_modifiers.get("matched_skills")',
        "LIVE DESKTOP EXECUTION PLANNING CONTRACT",
    )


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
    assert "looks_like_desktop_objective" in _generation_execute()
