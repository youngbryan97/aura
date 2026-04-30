"""Top-level Autonomous Architecture Governor orchestration."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from core.architect.behavior_fingerprint import BehaviorFingerprinter
from core.architect.code_graph import LiveArchitectureGraphBuilder
from core.architect.config import ASAConfig
from core.architect.ghost_boot import GhostBootRunner
from core.architect.models import ArchitecturalSmell, ArchitectureGraph, MutationTier, PromotionDecision, PromotionStatus, RefactorPlan
from core.architect.post_promotion_monitor import PostPromotionMonitor
from core.architect.proof_obligations import ProofVerifier
from core.architect.promotion_governor import PromotionGovernor
from core.architect.refactor_planner import RefactorPlanner, plan_from_dict, plan_to_dict
from core.architect.rollback_manager import RollbackManager
from core.architect.shadow_workspace import ShadowRun, ShadowWorkspaceManager
from core.architect.smell_detector import SmellDetector
from core.runtime.atomic_writer import atomic_write_text


class AutonomousArchitectureGovernor:
    """Audit, plan, shadow, prove, promote, monitor, and rollback ASA changes."""

    def __init__(self, config: ASAConfig | None = None):
        self.config = config or ASAConfig.from_env()
        self.graph_builder = LiveArchitectureGraphBuilder(self.config)
        self.smell_detector = SmellDetector(self.config)
        self.planner = RefactorPlanner(self.config)
        self.shadow_manager = ShadowWorkspaceManager(self.config)
        self.ghost_runner = GhostBootRunner(self.config)
        self.rollback_manager = RollbackManager(self.config)
        self.proof_verifier = ProofVerifier(self.config)
        self.promotion_governor = PromotionGovernor(self.config)
        self.monitor = PostPromotionMonitor(self.config)
        self.fingerprinter = BehaviorFingerprinter(self.config)

    def build_graph(self) -> ArchitectureGraph:
        return self.graph_builder.build(persist=True)

    def detect_smells(self, graph: ArchitectureGraph | None = None) -> list[ArchitecturalSmell]:
        return self.smell_detector.detect(graph or self.build_graph())

    def audit(self) -> dict[str, Any]:
        graph = self.build_graph()
        smells = self.detect_smells(graph)
        report = {
            "repo_root": str(self.config.repo_root),
            "graph_metrics": graph.metrics,
            "smell_count": len(smells),
            "smells_by_severity": _count_by(smells, "severity"),
            "smells_by_kind": _count_by(smells, "kind"),
            "top_smells": [asdict(smell) for smell in smells[:25]],
        }
        reports = self.config.artifacts / "reports"
        reports.mkdir(parents=True, exist_ok=True)
        atomic_write_text(reports / "audit-latest.json", json.dumps(report, indent=2, sort_keys=True, default=str))
        return report

    def plan(self, target: str) -> RefactorPlan:
        graph = self.build_graph()
        smells = self.detect_smells(graph)
        return self.planner.plan_for_target(target, graph, smells, persist=True)

    def shadow_run(self, plan: RefactorPlan) -> tuple[ShadowRun, Any, Any, Any]:
        shadow = self.shadow_manager.create(plan)
        ghost = self.ghost_runner.run(plan, shadow)
        rollback = self.rollback_manager.create_packet(plan, shadow) if plan.changed_files else None
        if rollback is not None:
            rollback = self.rollback_manager.dry_run(rollback)
        proof = self.proof_verifier.verify(
            plan,
            ghost,
            rollback,
            baseline_root=self.config.repo_root,
            candidate_root=Path(shadow.shadow_root),
        )
        return shadow, ghost, rollback, proof

    def promote(self, plan: RefactorPlan, shadow: ShadowRun, proof: Any, rollback: Any) -> PromotionDecision:
        if rollback is None:
            return PromotionDecision(
                run_id=shadow.run_id,
                plan_id=plan.id,
                status=PromotionStatus.REJECTED,
                reason="rollback packet missing",
                receipt_hash=getattr(proof, "decision_hash", ""),
            )
        decision = self.promotion_governor.promote(plan, shadow, proof, rollback)
        if decision.status is PromotionStatus.PROMOTED and plan.risk_tier >= MutationTier.T2_REFACTOR:
            baseline = self.fingerprinter.capture(root=self.config.repo_root, changed_files=plan.changed_files)
            self.monitor.arm(plan, rollback, baseline)
        elif decision.status is PromotionStatus.PROMOTED:
            self.monitor.arm(plan, rollback, None)
        return decision

    def auto(self, *, tier_max: MutationTier | str = MutationTier.T1_CLEANUP) -> dict[str, Any]:
        max_tier = MutationTier.parse(tier_max)
        graph = self.build_graph()
        smells = self.detect_smells(graph)
        plan = self.planner.find_auto_cleanup_plan(graph, smells)
        if plan is None:
            return {"status": "no_candidate", "smell_count": len(smells)}
        if plan.risk_tier > max_tier:
            return {"status": "candidate_above_tier", "plan": plan_to_dict(plan), "tier_max": max_tier.name}
        shadow, ghost, rollback, proof = self.shadow_run(plan)
        decision = self.promote(plan, shadow, proof, rollback)
        observation = None
        if decision.status is PromotionStatus.PROMOTED:
            observation = self.monitor.check_once(decision.run_id)
        return {
            "status": decision.status.value,
            "plan": plan_to_dict(plan),
            "run_id": shadow.run_id,
            "ghost_passed": ghost.passed,
            "proof_passed": proof.passed,
            "decision": asdict(decision),
            "rollback_packet": asdict(rollback) if rollback is not None else None,
            "observation": asdict(observation) if observation is not None else None,
        }

    def proposal(self, target: str) -> RefactorPlan:
        plan = self.plan(target)
        proposals = self.config.artifacts / "proposals"
        proposals.mkdir(parents=True, exist_ok=True)
        atomic_write_text(proposals / f"{plan.id}.json", json.dumps(plan_to_dict(plan), indent=2, sort_keys=True, default=str))
        return plan

    def load_plan(self, plan_id_or_path: str) -> RefactorPlan:
        return self.planner.load_plan(plan_id_or_path)

    def load_shadow_run(self, run_id: str) -> ShadowRun:
        return self.shadow_manager.load_run(run_id)

    def rollback(self, run_id: str) -> Any:
        return self.rollback_manager.restore(run_id)

    def monitor_status(self, *, run_id: str | None = None) -> Any:
        if run_id:
            return self.monitor.check_once(run_id)
        return self.monitor.latest()


def _count_by(smells: list[ArchitecturalSmell], attr: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for smell in smells:
        value = getattr(smell, attr)
        key = value.name if hasattr(value, "name") else str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def load_plan_from_run(config: ASAConfig, run_id: str) -> RefactorPlan:
    manifest_path = config.artifacts / "shadow_runs" / run_id / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return plan_from_dict(payload["plan"])
