from __future__ import annotations

from core.unity.unity_repair import UnityRepairPlanner
from core.unity.unity_state import FragmentationReport, TemporalWindow, UnityState


def test_projected_repair_improves_unity_without_faking_resolution():
    unity_state = UnityState(
        temporal=TemporalWindow(tick_id="tick_repair"),
        unity_score=0.32,
        fragmentation_score=0.68,
        level="fragmented",
        repair_needed=True,
        repair_reasons=["draft_conflict", "ownership_ambiguity"],
    )
    report = FragmentationReport(
        unity_id=unity_state.unity_id,
        fragmentation_score=0.68,
        level="fragmented",
        top_causes=[
            ("draft_conflict", 0.7, "conflicting interpretations remain active"),
            ("ownership_ambiguity", 0.45, "authorship is unclear"),
        ],
        safe_to_act=False,
        safe_to_self_report=True,
    )

    planner = UnityRepairPlanner()
    plan = planner.plan(unity_state, report)
    projected = planner.project(unity_state, plan)

    assert "preserve competing drafts and answer with qualified uncertainty" in plan.steps
    assert projected.unity_score > unity_state.unity_score
    assert projected.unity_score <= 0.72
    assert projected.repair_needed is True
