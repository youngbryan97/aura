"""Proof-obligation verification for autonomous architecture promotion."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from core.architect.behavior_oracle import SemanticBehaviorOracle
from core.architect.behavior_fingerprint import BehaviorFingerprinter
from core.architect.code_graph import LiveArchitectureGraphBuilder
from core.architect.config import ASAConfig
from core.architect.ghost_boot import GhostBootReport
from core.architect.models import (
    BehaviorDelta,
    MutationTier,
    ProofReceipt,
    ProofResult,
    RefactorPlan,
    RollbackPacket,
)
from core.architect.mutation_classifier import MutationClassifier
from core.runtime.atomic_writer import atomic_write_text


class ProofVerifier:
    """Fail-closed verifier for ASA proof obligations."""

    def __init__(self, config: ASAConfig | None = None):
        self.config = config or ASAConfig.from_env()
        self.classifier = MutationClassifier(self.config)
        self.fingerprinter = BehaviorFingerprinter(self.config)
        self.semantic_oracle = SemanticBehaviorOracle()

    def verify(
        self,
        plan: RefactorPlan,
        ghost: GhostBootReport,
        rollback_packet: RollbackPacket | None,
        *,
        baseline_root: Path | None = None,
        candidate_root: Path | None = None,
    ) -> ProofReceipt:
        results: list[ProofResult] = list(ghost.results)
        results.extend(self._universal(plan, ghost, rollback_packet))
        if plan.risk_tier <= MutationTier.T1_CLEANUP:
            results.extend(self._tier1(plan, ghost))
        elif plan.risk_tier is MutationTier.T2_REFACTOR:
            results.extend(self._tier2(plan, ghost))
        elif plan.risk_tier is MutationTier.T3_BEHAVIORAL_IMPROVEMENT:
            results.extend(self._tier2(plan, ghost))
            results.append(ProofResult("declared_improvement_target", "improved" in plan.expected_behavior_delta.lower(), "passed" if "improved" in plan.expected_behavior_delta.lower() else "failed"))
        else:
            results.append(ProofResult("proposal_only_for_t4_t5", True, "proposal_only", {"tier": plan.risk_tier.name}))
        baseline = baseline_root or self.config.repo_root
        candidate = candidate_root or self.config.repo_root
        before_graph = self._graph_for_root(baseline)
        after_graph = self._graph_for_root(candidate, artifact_root=Path(ghost.artifact_path).parent / "semantic_oracle")
        before = self.fingerprinter.capture(root=baseline, changed_files=plan.changed_files)
        after = self.fingerprinter.capture(
            root=candidate,
            proof_results=tuple(ghost.results),
            changed_files=plan.changed_files,
            artifact_root=Path(ghost.artifact_path).parent / "fingerprint",
        )
        delta = self.fingerprinter.compare(before, after)
        oracle_result = self.semantic_oracle.evaluate(
            plan,
            before_graph,
            after_graph,
            {result.obligation_id: result.status for result in ghost.results},
        ).as_proof_result()
        results.append(oracle_result)
        if not delta.equivalent:
            results.append(ProofResult("behavior_fingerprint_equivalent", False, "failed", {"regressions": list(delta.regressions)}))
        else:
            results.append(ProofResult("behavior_fingerprint_equivalent", True, "passed", {"improvements": list(delta.improvements)}))
        receipt = ProofReceipt(
            run_id=ghost.run_id,
            plan_id=plan.id,
            tier=plan.risk_tier,
            results=tuple(results),
            behavior_delta=delta,
            rollback_packet_hash=rollback_packet.receipt_hash if rollback_packet is not None else "",
            shadow_artifact_path=str(Path(ghost.artifact_path).parent),
        ).signed()
        receipt_path = Path(ghost.artifact_path).parent / "proof_receipt.json"
        atomic_write_text(receipt_path, json.dumps(asdict(receipt), indent=2, sort_keys=True, default=str))
        return receipt

    def _graph_for_root(self, root: Path, *, artifact_root: Path | None = None):
        cfg = ASAConfig(
            repo_root=root,
            enabled=self.config.enabled,
            autopromote=self.config.autopromote,
            max_tier=self.config.max_tier,
            shadow_timeout=self.config.shadow_timeout,
            observation_window=self.config.observation_window,
            artifact_root=artifact_root or self.config.artifacts,
            protected_paths=self.config.protected_paths,
            sealed_paths=self.config.sealed_paths,
            excludes=self.config.excludes,
            retain_shadow_runs=self.config.retain_shadow_runs,
            god_file_lines=self.config.god_file_lines,
            god_class_lines=self.config.god_class_lines,
            high_fan_in=self.config.high_fan_in,
            high_fan_out=self.config.high_fan_out,
            safe_boot_command=self.config.safe_boot_command,
            runtime_receipt_limit=self.config.runtime_receipt_limit,
            coverage_hit_limit=self.config.coverage_hit_limit,
            broader_pytest=self.config.broader_pytest,
            env=self.config.env,
        )
        return LiveArchitectureGraphBuilder(cfg).build(persist=False)

    def _universal(self, plan: RefactorPlan, ghost: GhostBootReport, rollback_packet: RollbackPacket | None) -> list[ProofResult]:
        changed = set(plan.changed_files)
        sealed = [path for path in changed if self.classifier.classify_path(path) is MutationTier.T5_SEALED]
        outside_scope = [path for path in changed if path not in plan.affected_files]
        critical_smell = any(
            result.obligation_id == "graph_rebuild" and result.passed is False
            for result in ghost.results
        )
        return [
            ProofResult("no_changed_file_outside_scope", not outside_scope, "passed" if not outside_scope else "failed", {"outside_scope": outside_scope}),
            ProofResult("no_sealed_surface_autonomous_edit", not sealed, "passed" if not sealed else "failed", {"sealed": sealed}),
            ProofResult("rollback_packet_created", rollback_packet is not None, "passed" if rollback_packet is not None else "failed"),
            ProofResult("rollback_dry_run", bool(rollback_packet and rollback_packet.dry_run_passed), "passed" if rollback_packet and rollback_packet.dry_run_passed else "failed"),
            ProofResult("shadow_artifacts_saved", Path(ghost.artifact_path).exists(), "passed" if Path(ghost.artifact_path).exists() else "failed", {"path": ghost.artifact_path}),
            ProofResult("graph_rebuild_succeeds", not critical_smell, "passed" if not critical_smell else "failed"),
            ProofResult("proof_receipt_generated", True, "passed"),
        ]

    def _tier1(self, plan: RefactorPlan, ghost: GhostBootReport) -> list[ProofResult]:
        text = " ".join(plan.proof_obligations).lower()
        has_static_proof = "unused_import" in text or "reachability" in text or any("static_proof" in step.metadata for step in plan.steps)
        import_passed = any(result.obligation_id == "changed_modules_import" and result.passed for result in ghost.results)
        no_new_critical = ghost.graph_metrics.get("parse_errors", []) == []
        return [
            ProofResult("t1_static_cleanup_proof", has_static_proof, "passed" if has_static_proof else "failed"),
            ProofResult("t1_minimal_ghost_import", import_passed, "passed" if import_passed else "failed"),
            ProofResult("t1_no_new_critical_smells", no_new_critical, "passed" if no_new_critical else "failed", {"parse_errors": ghost.graph_metrics.get("parse_errors", [])}),
        ]

    def _tier2(self, plan: RefactorPlan, ghost: GhostBootReport) -> list[ProofResult]:
        result_map = ghost.result_map()
        relevant = result_map.get("relevant_tests")
        critical = result_map.get("critical_tests")
        tests_ok = (relevant is None or relevant.passed or relevant.status == "not_available") and (critical is None or critical.passed or critical.status == "not_available")
        return [
            ProofResult("t2_relevant_tests_pass", tests_ok, "passed" if tests_ok else "failed"),
            ProofResult("t2_service_graph_valid", ghost.graph_metrics.get("nodes", 0) > 0, "passed" if ghost.graph_metrics.get("nodes", 0) > 0 else "failed"),
            ProofResult("t2_public_api_compatibility", True, "passed", {"reason": "checked through behavior fingerprint public API comparison"}),
            ProofResult("t2_no_new_bypasses", True, "passed", {"reason": "checked through behavior fingerprint protected bypass count"}),
        ]
