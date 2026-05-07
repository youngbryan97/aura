from __future__ import annotations

from typing import Any, Dict, Iterable

from .co_presence_graph import CoPresenceGraphSnapshot
from .unity_state import (
    BoundContent,
    FragmentationReport,
    ReconciledDraftSet,
    SelfWorldBinding,
    TemporalWindow,
    UnityState,
)

_CAUSE_EXPLANATIONS = {
    "temporal_discontinuity": "recent contents are not holding together across the rolling present",
    "phase_lag": "some subsystem snapshots are older than the active present",
    "draft_conflict": "multiple interpretations remain active and unresolved",
    "workspace_conflict": "co-present contents are pulling in incompatible directions",
    "affect_mismatch": "felt state and embodied strain are not lining up cleanly",
    "ownership_ambiguity": "self/world authorship is not sharply bounded",
    "memory_discontinuity": "continuity across active context and recall is thin",
    "goal_conflict": "multiple action pressures remain unresolved",
    "substrate_instability": "baseline coherence is degraded before narration begins",
}


class UnityMonitor:
    """Compute a durable UnityState and grounded fragmentation report."""

    def _affect_alignment(self, state: Any, contents: Iterable[BoundContent]) -> float:
        affect = getattr(state, "affect", None)
        valence = float(getattr(affect, "valence", 0.0) or 0.0)
        strain = float(affect.physiological_strain() if affect and hasattr(affect, "physiological_strain") else 0.0)
        affective_contents = [item for item in contents if item.modality == "affect"]
        if not affective_contents:
            return max(0.0, min(1.0, 1.0 - (strain * 0.5)))
        average_charge = sum(float(item.affective_charge or 0.0) for item in affective_contents) / max(1, len(affective_contents))
        charge_delta = min(1.0, abs(average_charge - valence))
        return max(0.0, min(1.0, 1.0 - (charge_delta * 0.6) - (strain * 0.3)))

    def _memory_continuity(self, state: Any, draft_set: ReconciledDraftSet) -> float:
        cognition = getattr(state, "cognition", None)
        rolling_summary = bool(getattr(cognition, "rolling_summary", ""))
        long_term = list(getattr(cognition, "long_term_memory", []) or [])
        contradiction_count = int(getattr(cognition, "contradiction_count", 0) or 0)

        score = 0.45
        if rolling_summary:
            score += 0.2
        if long_term:
            score += 0.15
        if getattr(cognition, "working_memory", None):
            score += 0.1
        score -= min(0.25, contradiction_count * 0.05)
        score -= draft_set.contradiction_score * 0.15
        return max(0.0, min(1.0, score))

    def _goal_conflict(self, state: Any, graph: CoPresenceGraphSnapshot, draft_set: ReconciledDraftSet) -> float:
        goals = list(getattr(getattr(state, "cognition", None), "active_goals", []) or [])
        if len(goals) <= 1:
            return min(1.0, draft_set.contradiction_score * 0.5)
        peripheral_slack = 1.0 - float(graph.metrics.get("focus_periphery_binding_strength", 0.0) or 0.0)
        return max(0.0, min(1.0, 0.2 + (len(goals) - 1) * 0.1 + peripheral_slack * 0.4))

    def _substrate_instability(self, state: Any) -> float:
        cognitive = getattr(state, "cognition", None)
        coherence = float(getattr(cognitive, "coherence_score", 1.0) or 1.0)
        phi = float(getattr(state, "phi", 0.0) or 0.0)
        return max(0.0, min(1.0, ((1.0 - coherence) * 0.6) + (max(0.0, 0.35 - phi) * 0.7)))

    def compute(
        self,
        state: Any,
        temporal: TemporalWindow,
        graph: CoPresenceGraphSnapshot,
        draft_set: ReconciledDraftSet,
        self_world: SelfWorldBinding,
        *,
        will_receipt_id: str | None = None,
        state_version: int | None = None,
    ) -> tuple[UnityState, FragmentationReport]:
        affect_alignment_score = self._affect_alignment(state, getattr(state, "_unity_contents", []) or [])
        temporal_score = max(
            0.0,
            min(
                1.0,
                float(temporal.continuity_from_previous or 0.0)
                * (1.0 - min(1.0, max(temporal.phase_lag.values(), default=0.0) / max(temporal.duration_s or 1.0, 1.0))),
            ),
        )
        cross_modal_score = max(
            0.0,
            min(
                1.0,
                (float(graph.metrics.get("largest_connected_component_ratio", 0.0)) * 0.55)
                + (float(graph.metrics.get("cross_modal_edge_density", 0.0)) * 0.45),
            ),
        )
        agency_score = max(
            0.0,
            min(
                1.0,
                (float(self_world.boundary_integrity or 0.0) * 0.35)
                + (float(self_world.ownership_confidence or 0.0) * 0.25)
                + (float(self_world.agency_score or 0.0) * 0.4),
            ),
        )
        memory_score = self._memory_continuity(state, draft_set)
        action_readiness = max(
            0.0,
            min(
                1.0,
                (
                    temporal_score
                    + cross_modal_score
                    + draft_set.consensus_score
                    + agency_score
                    + memory_score
                )
                / 5.0,
            ),
        )
        self_world_boundary_score = float(self_world.boundary_integrity or 0.0)
        draft_consensus_score = float(draft_set.consensus_score or 0.0)

        causes: Dict[str, float] = {
            "temporal_discontinuity": round(max(0.0, 1.0 - temporal_score), 4),
            "phase_lag": round(min(1.0, max(temporal.phase_lag.values(), default=0.0) / max(temporal.duration_s or 1.0, 1.0)), 4),
            "draft_conflict": round(float(draft_set.contradiction_score or 0.0), 4),
            "workspace_conflict": round(float(graph.metrics.get("conflict_density", 0.0) or 0.0), 4),
            "affect_mismatch": round(max(0.0, 1.0 - affect_alignment_score), 4),
            "ownership_ambiguity": round(max(0.0, 1.0 - float(self_world.ownership_confidence or 0.0)), 4),
            "memory_discontinuity": round(max(0.0, 1.0 - memory_score), 4),
            "goal_conflict": round(self._goal_conflict(state, graph, draft_set), 4),
            "substrate_instability": round(self._substrate_instability(state), 4),
        }

        weighted_fragmentation = (
            causes["temporal_discontinuity"] * 0.16
            + causes["phase_lag"] * 0.08
            + causes["draft_conflict"] * 0.2
            + causes["workspace_conflict"] * 0.12
            + causes["affect_mismatch"] * 0.1
            + causes["ownership_ambiguity"] * 0.12
            + causes["memory_discontinuity"] * 0.12
            + causes["goal_conflict"] * 0.05
            + causes["substrate_instability"] * 0.05
        )
        fragmentation_score = max(0.0, min(1.0, weighted_fragmentation))

        unity_score = max(
            0.0,
            min(
                1.0,
                (
                    temporal_score * 0.18
                    + self_world_boundary_score * 0.12
                    + cross_modal_score * 0.15
                    + draft_consensus_score * 0.15
                    + affect_alignment_score * 0.1
                    + agency_score * 0.1
                    + memory_score * 0.1
                    + action_readiness * 0.1
                ),
            ),
        )

        if unity_score >= 0.75:
            level = "coherent"
        elif unity_score >= 0.55:
            level = "strained"
        elif unity_score >= 0.35:
            level = "fragmented"
        else:
            level = "dissociated"

        top_causes = [
            (name, weight, _CAUSE_EXPLANATIONS[name])
            for name, weight in sorted(causes.items(), key=lambda item: item[1], reverse=True)
            if weight >= 0.12
        ][:4]
        repair_reasons = [name for name, _weight, _text in top_causes]
        repair_needed = level in {"fragmented", "dissociated"} or any(weight >= 0.35 for _name, weight, _text in top_causes)
        safe_to_act = level == "coherent" or (level == "strained" and fragmentation_score < 0.45)
        safe_to_self_report = level == "coherent" or bool(top_causes)

        summary = "Unity nominal."
        if top_causes:
            summary = "Fragmentation is driven by " + ", ".join(
                f"{name.replace('_', ' ')} ({weight:.2f})" for name, weight, _text in top_causes[:2]
            ) + "."

        unity_state = UnityState(
            temporal=temporal,
            contents=list(getattr(state, "_unity_contents", []) or []),
            draft_bindings=[draft_set.chosen] + list(draft_set.alternatives),
            global_focus_id=graph.focus_id,
            peripheral_content_ids=list(graph.peripheral_ids),
            self_world_boundary_score=round(self_world_boundary_score, 4),
            temporal_continuity_score=round(temporal_score, 4),
            cross_modal_coherence_score=round(cross_modal_score, 4),
            draft_consensus_score=round(draft_consensus_score, 4),
            affect_alignment_score=round(affect_alignment_score, 4),
            agency_ownership_score=round(agency_score, 4),
            memory_continuity_score=round(memory_score, 4),
            action_readiness_score=round(action_readiness, 4),
            fragmentation_score=round(fragmentation_score, 4),
            unity_score=round(unity_score, 4),
            level=level,
            repair_needed=repair_needed,
            repair_reasons=repair_reasons,
            will_receipt_id=will_receipt_id,
            state_version=state_version,
            metadata={
                "co_presence_metrics": dict(graph.metrics),
                "draft_commit_mode": draft_set.memory_commit_mode,
                "top_causes": top_causes,
                "self_world_binding": self_world.to_dict(),
            },
        )
        report = FragmentationReport(
            unity_id=unity_state.unity_id,
            fragmentation_score=round(fragmentation_score, 4),
            level=level,
            top_causes=top_causes,
            repair_recommendations=[_CAUSE_EXPLANATIONS[name] for name in repair_reasons[:3]],
            user_visible_summary=summary,
            safe_to_act=safe_to_act,
            safe_to_self_report=safe_to_self_report,
        )
        return unity_state, report
