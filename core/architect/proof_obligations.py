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
            # CP126 17be716a: this passed when expected_behavior_delta merely
            # CONTAINED the word "improved" — a declaration is not a
            # measurement. A T3 improvement claim needs a named target, a
            # baseline, and a measured direction; absent that, it is UNPROVEN.
            results.append(self._tier3_improvement_proof(plan))
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
        if plan.risk_tier >= MutationTier.T2_REFACTOR:
            # CP126 da90a3bf: these were CONSTRUCTED with passed=True citing
            # "checked through behavior fingerprint ..." logic that this method
            # never consulted. Decide them from the measured fingerprints.
            results.extend(self._fingerprint_derived_obligations(before, after, delta))
        # Authoritative non-LLM harness gate (LLM is advisory only)
        try:
            import asyncio

            from core.self_modification.safe_modification_harness import SafeModificationHarness
            harness = SafeModificationHarness(candidate)
            harness_coro = harness.run(list(plan.changed_files))
            try:
                harness_result = asyncio.run(harness_coro)
            except RuntimeError:
                # Already inside a running event loop — use nest_asyncio or thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    harness_result = pool.submit(asyncio.run, harness_coro).result(timeout=30)
            results.append(ProofResult(
                "safe_harness_gate",
                harness_result.passed,
                "passed" if harness_result.passed else "failed",
                {"checks": harness_result.checks, "errors": harness_result.errors[:5]},
            ))
        except (ImportError, RuntimeError, OSError) as exc:
            results.append(ProofResult("safe_harness_gate", False, "failed", {"error": str(exc)}))

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

    @staticmethod
    def _fingerprint_derived_obligations(before, after, delta) -> list[ProofResult]:
        """Public-API compatibility and no-new-bypasses, from measurement.

        Both were previously hard-coded passes that merely NAMED this
        comparison (CP126 da90a3bf).
        """
        changed_apis = tuple(getattr(after, "changed_public_apis", ()) or ())
        api_ok = not changed_apis
        before_bypasses = int(getattr(before, "protected_bypass_count", 0) or 0)
        after_bypasses = int(getattr(after, "protected_bypass_count", 0) or 0)
        no_new_bypasses = after_bypasses <= before_bypasses
        return [
            ProofResult(
                "t2_public_api_compatibility",
                api_ok,
                "passed" if api_ok else "failed",
                {"changed_public_apis": list(changed_apis[:8])},
            ),
            ProofResult(
                "t2_no_new_bypasses",
                no_new_bypasses,
                "passed" if no_new_bypasses else "failed",
                {"before": before_bypasses, "after": after_bypasses},
            ),
        ]

    @staticmethod
    def _tier3_improvement_proof(plan: RefactorPlan) -> ProofResult:
        """A T3 improvement must be MEASURED, not asserted in prose.

        The plan has to name the metric, its baseline, and its target so the
        claim can be checked against the behavior delta later. Wording alone
        is a CONJECTURE, and this receipt says so rather than passing.
        """
        evidence: dict[str, object] = {}
        target = None
        for step in plan.steps:
            metadata = getattr(step, "metadata", None) or {}
            candidate = metadata.get("improvement_target")
            if isinstance(candidate, dict):
                target = candidate
                break
        if not isinstance(target, dict):
            return ProofResult(
                "declared_improvement_target",
                False,
                "unproven_no_measured_target",
                {"reason": "no step declared an improvement_target with metric/baseline/goal"},
            )
        metric = str(target.get("metric") or "").strip()
        baseline = target.get("baseline")
        goal = target.get("goal")
        numeric = all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in (baseline, goal)
        )
        evidence = {"metric": metric, "baseline": baseline, "goal": goal}
        if not metric or not numeric:
            return ProofResult(
                "declared_improvement_target",
                False,
                "unproven_incomplete_target",
                evidence,
            )
        if float(goal) == float(baseline):
            return ProofResult(
                "declared_improvement_target", False, "unproven_no_effect_size", evidence
            )
        return ProofResult("declared_improvement_target", True, "passed", evidence)

    def _tier1(self, plan: RefactorPlan, ghost: GhostBootReport) -> list[ProofResult]:
        # CP126 a8434f1a: this passed when the obligation TEXT contained
        # "unused_import"/"reachability", or when step metadata merely had a
        # "static_proof" KEY — no analysis artifact was ever checked. The
        # proof now requires a step to carry an actual static-analysis result
        # naming the symbol it proved.
        static_evidence: list[dict[str, object]] = []
        for step in plan.steps:
            metadata = getattr(step, "metadata", None) or {}
            artifact = metadata.get("static_proof")
            if not isinstance(artifact, dict):
                continue
            analysis = str(artifact.get("analysis") or "").strip()
            symbol = str(artifact.get("symbol") or "").strip()
            if analysis and symbol and artifact.get("proved") is True:
                static_evidence.append(
                    {"target": step.target_path, "analysis": analysis, "symbol": symbol}
                )
        has_static_proof = bool(static_evidence)
        import_passed = any(result.obligation_id == "changed_modules_import" and result.passed for result in ghost.results)
        no_new_critical = ghost.graph_metrics.get("parse_errors", []) == []
        return [
            ProofResult(
                "t1_static_cleanup_proof",
                has_static_proof,
                "passed" if has_static_proof else "unproven_no_static_analysis_artifact",
                {"evidence": static_evidence[:8]},
            ),
            ProofResult("t1_minimal_ghost_import", import_passed, "passed" if import_passed else "failed"),
            ProofResult("t1_no_new_critical_smells", no_new_critical, "passed" if no_new_critical else "failed", {"parse_errors": ghost.graph_metrics.get("parse_errors", [])}),
        ]

    def _tier2(self, plan: RefactorPlan, ghost: GhostBootReport) -> list[ProofResult]:
        result_map = ghost.result_map()
        relevant = result_map.get("relevant_tests")
        critical = result_map.get("critical_tests")
        # CP126 da90a3bf: UNAVAILABLE tests were accepted as passing, and the
        # API-compatibility and no-new-bypass obligations were CONSTRUCTED with
        # passed=True citing logic that runs elsewhere. Absence of a check is
        # not a passed check — an unavailable test is UNPROVEN, and the two
        # fingerprint-derived obligations report their real state.
        unavailable = [
            name
            for name, result in (("relevant_tests", relevant), ("critical_tests", critical))
            if result is not None and not result.passed and result.status == "not_available"
        ]
        tests_ok = (relevant is None or relevant.passed) and (critical is None or critical.passed)
        results = [
            ProofResult(
                "t2_relevant_tests_pass",
                tests_ok,
                "passed" if tests_ok else ("unproven_tests_unavailable" if unavailable else "failed"),
                {"unavailable": unavailable},
            ),
            ProofResult("t2_service_graph_valid", ghost.graph_metrics.get("nodes", 0) > 0, "passed" if ghost.graph_metrics.get("nodes", 0) > 0 else "failed"),
        ]
        # The public-API and no-new-bypass obligations are DECIDED from the
        # behavior fingerprint comparison in verify(), not minted here.
        return results
