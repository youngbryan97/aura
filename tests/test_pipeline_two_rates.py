"""A user turn does not run 29 phases, and the code should be able to say so.

There is one ordered blueprint of 29 phases, and it gets described as the
per-turn cognitive sequence. ``AuraKernel._should_skip_priority_phase``
suppresses most of the expensive ones on a priority tick, so a healthy
foreground turn is closer to eleven plus conditional tool execution. The
rest are not dormant — MindTick runs ``kernel.tick(priority=False)`` over
the same kernel and commits the result to shared state.

The skip set used to live inline in the kernel where nothing else could read
it, which is why the description drifted: it had to be written from the
blueprint's length. These tests hold the mapping to the code.
"""
from __future__ import annotations

import inspect
import re

from core.runtime.pipeline_blueprint import (
    BACKGROUND_ONLY_PHASES,
    background_only_phase_names,
    foreground_phase_attributes,
    kernel_phase_attribute_order,
    phase_class_for_attribute,
    pipeline_rate_report,
)


def test_the_attribute_map_covers_every_phase_in_the_order():
    """An unmapped attribute would silently count as foreground."""
    unmapped = [
        attribute
        for attribute in kernel_phase_attribute_order()
        if phase_class_for_attribute(attribute) == attribute
    ]
    assert not unmapped, (
        f"these pipeline attributes have no phase-class mapping: {unmapped}. "
        "Unmapped means they can never match the skip set, so they would be "
        "reported as foreground whatever the kernel does."
    )


def test_the_foreground_is_a_strict_subset_and_much_smaller():
    report = pipeline_rate_report()
    assert report["blueprint_phases"] == len(kernel_phase_attribute_order())
    assert report["foreground_phases"] < report["blueprint_phases"]
    # If these ever converge, the two-rate architecture has gone away and the
    # docs describing it are wrong in the other direction.
    assert report["foreground_phases"] + report["background_only_phases"] == (
        report["blueprint_phases"]
    )


def test_the_expensive_phases_are_off_the_foreground_path():
    """The ones that make a turn slow, named individually.

    Listed rather than counted: a count would still pass if the set were
    swapped for a different eighteen.
    """
    foreground = {phase_class_for_attribute(a) for a in foreground_phase_attributes()}
    for expensive in (
        "PhiConsciousnessPhase",
        "CognitiveIntegrationPhase",
        "InferencePhase",
        "BondingPhase",
        "RepairPhase",
        "MemoryConsolidationPhase",
        "IdentityReflectionPhase",
        "InitiativeGenerationPhase",
        "ConsciousnessPhase",
        "SelfReviewPhase",
        "LearningPhase",
        "TrueEvolutionPhase",
    ):
        assert expensive not in foreground, (
            f"{expensive} is on the foreground path. It is one of the phases "
            "the priority tick exists to suppress; if that is deliberate, the "
            "latency budget and the docs both need revisiting."
        )


def test_the_phases_a_conversation_needs_are_on_the_foreground_path():
    """The other direction: protecting latency must not cost the answer."""
    foreground = set(foreground_phase_attributes())
    for essential in (
        "memory_retrieval_phase",   # continuity
        "affect_phase",             # state
        "executive_closure_phase",  # self-prediction and objective selection
        "routing_phase",
        "unity_phase",
        "response_phase",           # the reply itself
    ):
        assert essential in foreground, (
            f"{essential} is suppressed on a user-facing tick. The foreground "
            "path is latency-bounded, not stripped of cognition."
        )


def test_tool_execution_is_conditional_rather_than_always_on():
    """GodModeToolPhase runs on a priority tick only for SKILL/TASK intents."""
    report = pipeline_rate_report()
    assert "GodModeToolPhase" in report["conditional_on_priority_tick"]
    assert "GodModeToolPhase" in BACKGROUND_ONLY_PHASES
    assert "godmode_tools" not in foreground_phase_attributes()


def test_the_kernel_reads_the_shared_set_rather_than_its_own_copy():
    """Two copies of this list diverge, and the doc always loses."""
    from core.kernel.aura_kernel import AuraKernel

    source = inspect.getsource(AuraKernel._should_skip_priority_phase)
    assert "BACKGROUND_ONLY_PHASES" in source
    assert '"PhiConsciousnessPhase"' not in source, (
        "the kernel has grown its own copy of the skip set again"
    )


def test_the_report_says_what_it_means():
    report = pipeline_rate_report()
    assert "not 29 phases per turn" in report["note"]
    assert set(report["suppressed_on_priority_tick"]) <= set(BACKGROUND_ONLY_PHASES)
    assert len(report["suppressed_on_priority_tick"]) == len(
        background_only_phase_names()
    )


def test_architecture_md_names_the_phases_the_report_names():
    """The page and the code, held to the same list.

    ARCHITECTURE.md §2 is what anyone reads to learn what a turn runs, and it
    had been written from the blueprint rather than from the report — which is
    how it came to name a `Legacy` phase. `LegacyPhase` is the orchestrator
    bridge in core/kernel/bridge.py; it is in BACKGROUND_ONLY_PHASES, where
    its name can never match a pipeline phase, and from there it reached the
    page as a nineteenth thing a turn suppresses.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    page = (root / "ARCHITECTURE.md").read_text(errors="replace")
    marker = "Suppressed on a user-facing tick"
    assert marker in page, "ARCHITECTURE.md §2 no longer names the suppressed set"
    start = page.index(marker)
    listed = page[start:].split(".\n\n", 1)[0]

    report = pipeline_rate_report()
    for phase in report["suppressed_on_priority_tick"]:
        assert phase in listed, f"ARCHITECTURE.md §2 omits {phase}"

    count_match = re.search(r"\((\d+) phases", listed)
    assert count_match, "ARCHITECTURE.md §2 no longer states how many are suppressed"
    counted = int(count_match.group(1))
    assert counted == report["background_only_phases"], (
        f"ARCHITECTURE.md says {counted} suppressed phases, "
        f"the report says {report['background_only_phases']}"
    )

    foreground = page[page.index("A healthy foreground turn is") :][:200]
    assert "eleven phases" in foreground, (
        "the foreground count moved; ARCHITECTURE.md §2 still says eleven"
    )
    assert report["foreground_phases"] == 11
