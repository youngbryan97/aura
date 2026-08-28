"""A capability gap Aura notices is a capability gap she remembers.

The two halves were built and never joined. Gaps are recognised in the
self-improvement loop as signals of kind capability_gap, tool_gap,
missing_tool and unmet_affordance. Counting them, and forging a skill for one
seen often enough, lives in the skill synthesizer, whose ``log_gap`` had no
caller anywhere in the running system — only in its own test. So the forge's
tests passed over a path nothing could reach, and every gap she correctly
noticed was forgotten as soon as she noticed it.

Meanwhile the autonomy score awarded a fifth of a point, and the milestone
"skill_synthesis_active", for that same component being registered in the
service container.
"""

from __future__ import annotations

import core.evolution.evolution_orchestrator as evolution
from core.agi.skill_synthesizer import SkillSynthesizer
from core.learning.recursive_self_improvement import (
    ImprovementSignal,
    RecursiveSelfImprovementLoop,
)


def _gap(task: str) -> ImprovementSignal:
    return ImprovementSignal(
        source="skill_dispatch",
        kind="capability_gap",
        evidence={"task": task},
    )


def test_the_same_gap_twice_counts_twice() -> None:
    """The premise the ledger is built on: a gap can be recurring."""

    ledger = SkillSynthesizer()
    for _ in range(2):
        ledger.log_gap("convert a scanned invoice into a ledger entry", "capability_gap")
    assert ledger.get_status()["gap_count"] == 1
    assert ledger._gap_counts["convert a scanned invoice into a ledger entry"] == 2


def test_two_different_gaps_do_not_collapse_into_one() -> None:
    """Keyed on what was missing, not on the signal's kind.

    Reading the kind or the metric would give every gap the same key —
    "capability" for all of them — and a ledger built to notice the same gap
    twice would see one gap forever.
    """

    ledger = SkillSynthesizer()
    ledger.log_gap("convert a scanned invoice into a ledger entry", "capability_gap")
    ledger.log_gap("read a pressure gauge from a photograph", "capability_gap")
    assert ledger.get_status()["gap_count"] == 2


def test_the_loop_actually_calls_the_ledger() -> None:
    """The wiring itself, which is the thing that was missing.

    Asserted on the call, not on a mock's convenience: a real ledger goes in
    and has to come out holding the gap.
    """

    ledger = SkillSynthesizer()
    import core.agi.skill_synthesizer as synth_module

    original = synth_module.get_skill_synthesizer
    synth_module.get_skill_synthesizer = lambda: ledger
    try:
        RecursiveSelfImprovementLoop._remember_the_gaps(
            [_gap("convert a scanned invoice into a ledger entry")]
        )
    finally:
        synth_module.get_skill_synthesizer = original

    assert ledger.get_status()["gap_count"] == 1


def test_registration_is_not_use() -> None:
    """A score anybody can raise by constructing an object measures the constructor."""

    fresh = SkillSynthesizer()
    assert evolution._has_actually_seen_a_gap(fresh) is False

    fresh.log_gap("convert a scanned invoice into a ledger entry", "capability_gap")
    assert evolution._has_actually_seen_a_gap(fresh) is True
