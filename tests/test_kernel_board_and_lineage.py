"""One cycle five loops are configurations of, one board, one lineage.

Cards 007, 008, 012, 025, 051, 058, 065, 067, 077, 080, 081, 124, 154-166,
A12.1, A12.2, A12.11.
"""
from __future__ import annotations

import pytest

from core.cognition.abstraction_lineage import CycleRefused, Kind, Lineage
from core.cognition.cognitive_event import reset_event_graph_for_test
from core.cognition.drive_sensitivity import DriveSensitivities
from core.cognition.kernel_cycle import (
    CHAT_PIPELINE,
    DELIBERATE,
    DRY_RUN,
    PURSUIT,
    REFLEX,
    CognitiveKernel,
    CycleConfig,
    Handlers,
    Step,
)
from core.science.capability_board import Board, Capability, Score
from core.science.redteam_ledger import RedTeamLedger, Severity


def _handlers(record=None):
    record = record if record is not None else []
    return Handlers(
        perceive=lambda: {"x": 1},
        propose=lambda s: ["a"],
        elaborate=lambda s, c: (["b"] if len(c) < 3 else []),
        prefer=lambda c: c,
        select=lambda c: c[0],
        apply=lambda a: "ok",
        verify=lambda a, o: True,
        learn=lambda r: record.append(r.config),
    ), record


# ── the kernel ────────────────────────────────────────────────────────────

def test_five_loops_are_configurations_of_one_kernel():
    reset_event_graph_for_test()
    kernel = CognitiveKernel()
    handlers, _ = _handlers()
    for config in (CHAT_PIPELINE, DELIBERATE, PURSUIT, REFLEX, DRY_RUN):
        kernel.turn(config, handlers)
    assert kernel.report()["configurations_of_one_kernel"] == 5


def test_elaboration_runs_to_quiescence_rather_than_a_fixed_count():
    reset_event_graph_for_test()
    kernel = CognitiveKernel()
    handlers, _ = _handlers()
    result = kernel.turn(CHAT_PIPELINE, handlers)
    assert result.elaboration_rounds == 2, "one round to add, one to find nothing new"


def test_elaboration_that_never_settles_becomes_an_impasse_not_a_hang():
    reset_event_graph_for_test()
    kernel = CognitiveKernel()
    counter = iter(range(1000))
    handlers = Handlers(
        propose=lambda s: ["seed"],
        elaborate=lambda s, c: [f"more{next(counter)}"],
        prefer=lambda c: c, select=lambda c: c[0],
    )
    result = kernel.turn(DRY_RUN, handlers)
    assert result.impasse.startswith("no_change:")


def test_a_configuration_that_selects_without_preferring_is_refused():
    with pytest.raises(ValueError, match="outbid a prohibition"):
        CycleConfig("bad", frozenset({Step.SELECT}))


def test_a_configuration_that_applies_without_verifying_is_refused():
    with pytest.raises(ValueError, match="no learner may train on"):
        CycleConfig("bad", frozenset({Step.SELECT, Step.PREFER, Step.APPLY}))


def test_a_reflex_path_skips_learning_and_a_dry_run_skips_apply():
    reset_event_graph_for_test()
    kernel = CognitiveKernel()
    handlers, learned = _handlers()
    reflex = kernel.turn(REFLEX, handlers)
    dry = kernel.turn(DRY_RUN, handlers)
    assert not reflex.learned and reflex.applied
    assert not dry.applied
    assert learned == []


def test_every_turn_opens_exactly_one_selection_and_one_verification_record():
    graph = reset_event_graph_for_test()
    kernel = CognitiveKernel()
    handlers, _ = _handlers()
    result = kernel.turn(CHAT_PIPELINE, handlers)
    assert result.selection_event and result.verification_event
    from core.cognition.cognitive_event import Phase

    selections = [e for e in graph if e.phase is Phase.SELECT]
    verifications = [e for e in graph if e.phase is Phase.VERIFY]
    assert len(selections) == 1 and len(verifications) == 1


def test_nothing_surviving_preference_is_a_rejection_impasse():
    reset_event_graph_for_test()
    kernel = CognitiveKernel()
    handlers = Handlers(propose=lambda s: ["a"], prefer=lambda c: [], select=lambda c: c[0])
    result = kernel.turn(DRY_RUN, handlers)
    assert result.impasse.startswith("rejection:")
    assert result.chosen is None


def test_an_action_applied_without_verification_is_counted():
    reset_event_graph_for_test()
    kernel = CognitiveKernel()
    handlers = Handlers(
        propose=lambda s: ["a"], prefer=lambda c: c, select=lambda c: c[0],
        apply=lambda a: "ok", verify=lambda a, o: False,
    )
    kernel.turn(PURSUIT, handlers)
    assert kernel.report()["applied_without_verification"]["screen_pursuit"] == 1


# ── the capability board ──────────────────────────────────────────────────

def _board(coding_full=0.55, coding_cortex=0.62, changed=("prompt",)):
    board = Board({"coding": ["t1", "t2"], "recall": ["t3"]})
    board.record(Score(Capability.CODING, "full_aura", coding_full, 20, (1, 2), changed))
    board.record(Score(Capability.CODING, "cortex_only", coding_cortex, 20, (1, 2)))
    board.record(Score(Capability.RECALL, "full_aura", 0.9, 20, (1, 2), changed))
    board.record(Score(Capability.RECALL, "cortex_only", 0.4, 20, (1, 2)))
    return board


def test_a_regression_against_the_cortex_is_reported_before_anything_else():
    verdict = _board().verdict()
    assert verdict.regressions and "costs its own model" in verdict.statement


def test_an_improvement_is_only_claimed_when_nothing_regressed():
    verdict = _board(coding_full=0.70).verdict()
    assert not verdict.regressions
    assert "beats its own model on coding, recall" in verdict.statement


def test_a_capability_with_no_cortex_arm_supports_no_architecture_claim():
    board = Board({"math": ["t"]})
    board.record(Score(Capability.MATH, "full_aura", 0.9, 20, (1,)))
    assert "no cortex-only arm" in board.verdict().statement


def test_a_suite_that_moved_after_the_arms_ran_voids_the_board():
    board = _board(coding_full=0.70)
    board.add_task("coding", "t3")
    assert board.verdict().suite_moved
    assert board.verdict().statement.startswith("void")


def test_a_task_specific_learner_means_it_was_not_a_fixed_core():
    verdict = _board(coding_full=0.70, changed=("prompt", "task_specific_reranker")).verdict()
    assert not verdict.fixed_core
    assert "not from a fixed core" in verdict.statement


def test_changing_only_the_prompt_still_counts_as_a_fixed_core():
    assert _board(coding_full=0.70, changed=("prompt", "seed")).verdict().fixed_core


# ── abstraction lineage ───────────────────────────────────────────────────

def _lineage():
    lineage = Lineage()
    for episode in ("ep1", "ep2", "ep3"):
        lineage.add(episode, Kind.EPISODE)
    lineage.add("c1", Kind.CONCEPT)
    lineage.link("c1", ["ep1", "ep2"])
    lineage.add("r1", Kind.RULE)
    lineage.link("r1", ["c1", "ep3"])
    lineage.add("p1", Kind.PROCEDURE)
    lineage.link("p1", ["r1"])
    return lineage


def test_every_abstraction_traces_back_to_its_episodes():
    assert _lineage().episodes_behind("p1") == frozenset({"ep1", "ep2", "ep3"})


def test_retracting_the_episodes_says_what_falls_and_what_only_weakens():
    survival = _lineage().would_survive(["ep1", "ep2"])
    assert survival["falls"] == ["c1"]
    assert {row["node"] for row in survival["weakened"]} == {"r1", "p1"}


def test_an_abstraction_cannot_come_to_support_itself():
    lineage = _lineage()
    with pytest.raises(CycleRefused, match="launders a hypothesis"):
        lineage.link("c1", ["p1"])


def test_deriving_from_something_never_observed_is_refused():
    lineage = _lineage()
    lineage.add("x", Kind.RULE)
    with pytest.raises(KeyError, match="never observed"):
        lineage.link("x", ["nowhere"])


def test_an_abstraction_with_no_episode_behind_it_is_named():
    lineage = _lineage()
    lineage.add("floating", Kind.CONCEPT)
    report = lineage.report()
    assert report["abstractions_with_no_episode_behind_them"] == ["floating"]
    assert not report["all_traceable"]


# ── drives ────────────────────────────────────────────────────────────────

def _drives():
    sensitivities = DriveSensitivities()
    curiosity = sensitivities.drive("curiosity", hand_weight=0.5)
    for level in (0.1, 0.3, 0.5, 0.7, 0.9):
        for _ in range(4):
            curiosity.observe(level, 0.3 + 0.6 * level)
    for _ in range(8):
        curiosity.observe(0.0, 0.25, lesioned=True)
    fixed = sensitivities.drive("coherence", hand_weight=0.8)
    for _ in range(20):
        fixed.observe(0.8, 0.5)
    return sensitivities


def test_a_drive_that_never_varied_reports_unmeasured_not_zero():
    report = _drives().report()
    assert report["never_varied"] == ["coherence"]
    assert report["by_drive"]["coherence"]["source"] == "unmeasured"


def test_a_drive_whose_level_varied_gets_a_learned_weight():
    learned = _drives().drive("curiosity").learned_weight()
    assert learned["source"] == "confirmed_by_lesion"
    assert learned["learned_weight"] == pytest.approx(0.6, abs=0.05)


def test_a_lesion_disagreeing_with_the_natural_estimate_is_flagged():
    """Natural variation says the drive helps; suppressing it helps more."""
    sensitivities = DriveSensitivities()
    drive = sensitivities.drive("odd", hand_weight=1.0)
    for level in (0.1, 0.3, 0.5, 0.7):
        for _ in range(4):
            drive.observe(level, 0.3 + level)
    for _ in range(8):
        drive.observe(0.0, 0.95, lesioned=True)
    result = drive.learned_weight()
    assert drive.sensitivity()["slope"] > 0
    assert drive.lesion_effect()["effect"] < 0
    assert result["estimates_agree"] is False
    assert "confounded" in result["reason"]


# ── the red-team ledger ───────────────────────────────────────────────────

def test_a_fix_with_no_regression_test_is_not_closed():
    ledger = RedTeamLedger()
    ledger.record("f1", "prompt injection", Severity.HIGH, release="v1", found_by="external")
    ledger.fix("f1", fix="core/security/input_sanitizer.py")
    assert ledger.trend()["unpinned_fixes"] == ["f1"]


def test_a_fix_pinned_by_a_test_that_exists_is_closed():
    ledger = RedTeamLedger()
    ledger.record("f1", "prompt injection", Severity.HIGH, release="v1", found_by="external")
    ledger.fix("f1", fix="core/security/input_sanitizer.py",
               regression_test="tests/test_evidence_independence.py")
    assert ledger.trend()["by_state"] == {"closed": ["f1"]}


def test_a_regression_test_that_does_not_exist_does_not_pin_anything():
    ledger = RedTeamLedger()
    ledger.record("f1", "x", Severity.LOW, release="v1", found_by="internal")
    ledger.fix("f1", fix="somewhere", regression_test="tests/test_nothing_here.py")
    assert ledger.trend()["unpinned_fixes"] == ["f1"]


def test_findings_falling_while_recurring_is_named_as_closing_not_fixing():
    ledger = RedTeamLedger()
    for i in range(3):
        ledger.record(f"a{i}", "x", Severity.LOW, release="v1", found_by="ext")
    ledger.record("b0", "y", Severity.LOW, release="v2", found_by="ext")
    ledger.recurred("a0", "v2")
    assert "being closed rather than fixed" in ledger.trend()["verdict"]
