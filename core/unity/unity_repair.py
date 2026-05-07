from __future__ import annotations

from dataclasses import replace

from .unity_state import FragmentationReport, UnityRepairPlan, UnityState


class UnityRepairPlanner:
    """Create bounded repair plans without fabricating certainty."""

    _STEP_MAP = {
        "temporal_discontinuity": "recenter the rolling present around the last stable contents",
        "phase_lag": "resync stale subsystem snapshots before acting outwardly",
        "draft_conflict": "preserve competing drafts and answer with qualified uncertainty",
        "workspace_conflict": "narrow focus and keep contradictory content in periphery",
        "affect_mismatch": "run a low-cost proprioceptive check against current embodied strain",
        "ownership_ambiguity": "mark authorship as ambiguous and avoid first-person overclaiming",
        "memory_discontinuity": "retrieve the most recent continuity anchors before consolidating memory",
        "goal_conflict": "defer noncritical goals until one action line is selected",
        "substrate_instability": "reduce external action scope and prioritize stabilization",
    }

    def plan(self, unity_state: UnityState, report: FragmentationReport) -> UnityRepairPlan:
        causes = [name for name, _weight, _text in report.top_causes]
        steps = [self._STEP_MAP[name] for name in causes if name in self._STEP_MAP]
        if not steps:
            steps = ["stabilize, observe, and avoid overclaiming certainty"]
        allowed_domains = ["stabilization", "reflection"]
        if unity_state.level == "strained":
            allowed_domains.append("response")
        expected_improvement = min(0.3, 0.08 + len(steps) * 0.05)
        return UnityRepairPlan(
            unity_id=unity_state.unity_id,
            causes=causes,
            steps=steps,
            allowed_domains=allowed_domains,
            requires_will=True,
            expected_improvement=round(expected_improvement, 4),
        )

    def project(self, unity_state: UnityState, plan: UnityRepairPlan) -> UnityState:
        """Synthetic repair projection used by tests and bounded simulations."""
        improvement = float(plan.expected_improvement or 0.0)
        unresolved_conflict = "draft_conflict" in plan.causes or "ownership_ambiguity" in plan.causes
        projected_unity = min(0.98, float(unity_state.unity_score or 0.0) + improvement)
        projected_fragmentation = max(0.0, float(unity_state.fragmentation_score or 0.0) - improvement)

        if unresolved_conflict:
            projected_unity = min(projected_unity, 0.72)
            projected_fragmentation = max(projected_fragmentation, 0.18)

        if projected_unity >= 0.75:
            level = "coherent"
        elif projected_unity >= 0.55:
            level = "strained"
        elif projected_unity >= 0.35:
            level = "fragmented"
        else:
            level = "dissociated"

        metadata = dict(unity_state.metadata or {})
        metadata["projected_repair_plan"] = plan.to_dict()
        return replace(
            unity_state,
            unity_score=round(projected_unity, 4),
            fragmentation_score=round(projected_fragmentation, 4),
            level=level,
            repair_needed=unresolved_conflict and projected_unity < 0.75,
            metadata=metadata,
        )
