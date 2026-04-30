"""Before/after behavior fingerprints for architecture promotion."""
from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Any

from core.architect.code_graph import LiveArchitectureGraphBuilder
from core.architect.config import ASAConfig
from core.architect.models import BehaviorDelta, BehaviorFingerprint, ProofResult
from core.architect.smell_detector import SmellDetector


class BehaviorFingerprinter:
    """Collect deterministic architecture and proof metrics."""

    def __init__(self, config: ASAConfig | None = None):
        self.config = config or ASAConfig.from_env()

    def capture(
        self,
        *,
        root: Path | None = None,
        proof_results: tuple[ProofResult, ...] = (),
        changed_files: tuple[str, ...] = (),
        artifact_root: Path | None = None,
    ) -> BehaviorFingerprint:
        cfg = self.config
        if root is not None:
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
        graph = LiveArchitectureGraphBuilder(cfg).build(persist=False)
        smells = SmellDetector(cfg).detect(graph)
        smell_counts = Counter(smell.severity.name for smell in smells)
        smell_kinds = Counter(smell.kind for smell in smells)
        results = {result.obligation_id: result for result in proof_results}
        compile_status = _proof_status(results, "syntax")
        import_status = _proof_status(results, "changed_modules_import")
        boot_status = _proof_status(results, "safe_boot")
        public_apis = tuple(
            sorted(
                node.qualified_name
                for node in graph.nodes.values()
                if node.path in changed_files and node.kind in {"class", "function", "async_function"} and not node.name.startswith("_")
            )
        )
        service_regs = tuple(
            sorted(
                node.path
                for node in graph.nodes.values()
                if node.kind == "file" and "service_container_register" in node.metadata.get("effects", ())
            )
        )
        receipt_kinds = graph.metrics.get("runtime_receipts_by_kind", {})
        optional = {
            "phi": "not_available",
            "gwt": "not_available",
            "affect_neurochemical": "not_available",
            "memory_consolidation_receipts": receipt_kinds.get("memory_write", 0),
            "tool_capability_receipts": {
                "tool_execution": receipt_kinds.get("tool_execution", 0),
                "capability": receipt_kinds.get("capability", 0),
            },
            "identity_consistency": "not_available",
            "response_fingerprints": "not_available",
            "runtime_receipts_by_kind": receipt_kinds,
            "runtime_receipt_paths": graph.metrics.get("runtime_receipt_paths", 0),
        }
        fingerprint_id = hashlib.sha256(
            f"{cfg.repo_root}:{graph.metrics.get('nodes')}:{graph.metrics.get('edges')}:{changed_files}".encode("utf-8")
        ).hexdigest()[:16]
        return BehaviorFingerprint(
            id=fingerprint_id,
            root=str(cfg.repo_root),
            graph_metrics=graph.metrics,
            smell_counts={**{key: int(value) for key, value in smell_counts.items()}, "kinds": dict(smell_kinds)},
            import_cycle_count=smell_kinds.get("import_cycle", 0),
            god_file_count=smell_kinds.get("god_file", 0),
            broad_exception_count=smell_kinds.get("broad_exception_cluster", 0),
            protected_bypass_count=smell_kinds.get("state_write_bypass", 0) + smell_kinds.get("tool_authority_bypass", 0),
            tests={
                key: result.evidence
                for key, result in results.items()
                if "tests" in key or key in {"critical_tests", "relevant_tests", "broader_pytest"}
            },
            compile_status=compile_status,
            import_status=import_status,
            boot_status=boot_status,
            changed_public_apis=public_apis,
            service_registrations=service_regs,
            authority_path_checks={"protected_bypass_count": smell_kinds.get("tool_authority_bypass", 0), "critical": smell_kinds.get("tool_authority_bypass", 0) == 0},
            memory_state_write_checks={
                "memory_bypass_count": smell_kinds.get("memory_write_bypass", 0),
                "state_bypass_count": smell_kinds.get("state_write_bypass", 0),
                "critical": smell_kinds.get("memory_write_bypass", 0) == 0 and smell_kinds.get("state_write_bypass", 0) == 0,
            },
            latency_resource={
                result.obligation_id: result.evidence
                for result in proof_results
                if "duration_s" in result.evidence or "timed_out" in result.evidence
            },
            optional_runtime_metrics=optional,
        )

    def compare(self, before: BehaviorFingerprint, after: BehaviorFingerprint) -> BehaviorDelta:
        regressions: list[str] = []
        improvements: list[str] = []
        if after.import_cycle_count > before.import_cycle_count:
            regressions.append("import cycles increased")
        if after.god_file_count > before.god_file_count:
            regressions.append("god-file count increased")
        if after.broad_exception_count > before.broad_exception_count:
            regressions.append("broad exception count increased")
        if after.protected_bypass_count > before.protected_bypass_count:
            regressions.append("protected bypass count increased")
        if after.import_cycle_count < before.import_cycle_count:
            improvements.append("import cycles decreased")
        if after.broad_exception_count < before.broad_exception_count:
            improvements.append("broad exception count decreased")
        before_public = set(before.changed_public_apis)
        after_public = set(after.changed_public_apis)
        removed_public = before_public - after_public
        if removed_public:
            regressions.append(f"public API removed: {sorted(removed_public)[:5]}")
        equivalent = not regressions
        return BehaviorDelta(
            equivalent=equivalent,
            improved=equivalent and bool(improvements),
            regressions=tuple(regressions),
            improvements=tuple(improvements),
            details={
                "before": {
                    "cycles": before.import_cycle_count,
                    "god_files": before.god_file_count,
                    "broad_exceptions": before.broad_exception_count,
                    "protected_bypass": before.protected_bypass_count,
                },
                "after": {
                    "cycles": after.import_cycle_count,
                    "god_files": after.god_file_count,
                    "broad_exceptions": after.broad_exception_count,
                    "protected_bypass": after.protected_bypass_count,
                },
            },
        )


def _proof_status(results: dict[str, ProofResult], key: str) -> dict[str, Any]:
    result = results.get(key)
    if result is None:
        return {"status": "not_available"}
    return {"status": result.status, "passed": result.passed, "evidence": result.evidence}
