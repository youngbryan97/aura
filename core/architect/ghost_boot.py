"""Bounded ghost-boot runner for ASA candidates."""
from __future__ import annotations

import ast
import json
import py_compile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from core.architect.code_graph import LiveArchitectureGraphBuilder
from core.architect.config import ASAConfig
from core.architect.models import ProofResult, RefactorPlan
from core.architect.shadow_workspace import ShadowRun, ShadowWorkspaceManager, python_executable
from core.runtime.atomic_writer import atomic_write_text


@dataclass(frozen=True)
class GhostBootReport:
    run_id: str
    passed: bool
    results: tuple[ProofResult, ...]
    graph_metrics: dict[str, Any]
    artifact_path: str
    started_at: float = field(default_factory=time.time)

    def result_map(self) -> dict[str, ProofResult]:
        return {result.obligation_id: result for result in self.results}


class GhostBootRunner:
    """Run deterministic proof commands inside a shadow workspace."""

    def __init__(self, config: ASAConfig | None = None):
        self.config = config or ASAConfig.from_env()

    def run(self, plan: RefactorPlan, shadow: ShadowRun) -> GhostBootReport:
        manager = ShadowWorkspaceManager(self.config)
        results: list[ProofResult] = []
        results.append(self._compile_changed(shadow))
        results.append(self._import_changed(manager, shadow))
        graph_result, graph_metrics = self._rebuild_graph(shadow)
        results.append(graph_result)
        results.extend(self._architecture_quality_tests(manager, shadow))
        results.extend(self._relevant_tests(manager, shadow))
        results.extend(self._critical_tests(manager, shadow))
        if self.config.broader_pytest:
            results.append(self._broader_pytest(manager, shadow))
        results.append(self._safe_boot_check(manager, shadow))
        results.append(self._minimal_runtime_simulation(manager, shadow))
        passed = all(result.passed or result.status == "BOOT_HARNESS_UNAVAILABLE" for result in results)
        report = GhostBootReport(
            run_id=shadow.run_id,
            passed=passed,
            results=tuple(results),
            graph_metrics=graph_metrics,
            artifact_path=str(Path(shadow.artifact_dir) / "ghost_boot.json"),
        )
        atomic_write_text(report.artifact_path, json.dumps(asdict(report), indent=2, sort_keys=True, default=str))
        return report

    def _compile_changed(self, shadow: ShadowRun) -> ProofResult:
        errors: list[str] = []
        for rel in shadow.changed_files:
            path = Path(shadow.shadow_root) / rel
            if not path.exists() or not rel.endswith(".py"):
                continue
            try:
                source = path.read_text(encoding="utf-8")
                ast.parse(source, filename=rel)
                py_compile.compile(str(path), doraise=True)
            except (SyntaxError, py_compile.PyCompileError, UnicodeDecodeError, OSError) as exc:
                errors.append(f"{rel}:{exc}")
        return ProofResult(
            obligation_id="syntax",
            passed=not errors,
            status="passed" if not errors else "failed",
            evidence={"errors": errors, "changed_files": list(shadow.changed_files)},
        )

    def _import_changed(self, manager: ShadowWorkspaceManager, shadow: ShadowRun) -> ProofResult:
        modules = [
            Path(rel).with_suffix("").as_posix().replace("/", ".")
            for rel in shadow.changed_files
            if rel.endswith(".py") and Path(rel).name != "__init__.py"
        ]
        if not modules:
            return ProofResult("changed_modules_import", True, "passed", {"modules": []})
        script = "import importlib\n" + "\n".join(f"importlib.import_module({module!r})" for module in modules) + "\nprint('IMPORT_OK')\n"
        result = manager.run_command(shadow, (python_executable(), "-B", "-c", script), timeout=min(self.config.shadow_timeout, 20.0))
        return ProofResult(
            obligation_id="changed_modules_import",
            passed=result.exit_code == 0 and not result.timed_out,
            status="passed" if result.exit_code == 0 and not result.timed_out else "failed",
            evidence={"modules": modules, "exit_code": result.exit_code, "stdout": result.stdout, "stderr": result.stderr, "timed_out": result.timed_out},
        )

    def _rebuild_graph(self, shadow: ShadowRun) -> tuple[ProofResult, dict[str, Any]]:
        shadow_config = ASAConfig(
            repo_root=Path(shadow.shadow_root),
            enabled=self.config.enabled,
            autopromote=self.config.autopromote,
            max_tier=self.config.max_tier,
            shadow_timeout=self.config.shadow_timeout,
            observation_window=self.config.observation_window,
            artifact_root=Path(shadow.artifact_dir) / "graph_artifacts",
            protected_paths=self.config.protected_paths,
            sealed_paths=self.config.sealed_paths,
            excludes=self.config.excludes,
            retain_shadow_runs=self.config.retain_shadow_runs,
            god_file_lines=self.config.god_file_lines,
            god_class_lines=self.config.god_class_lines,
            high_fan_in=self.config.high_fan_in,
            high_fan_out=self.config.high_fan_out,
            safe_boot_command=self.config.safe_boot_command,
            broader_pytest=self.config.broader_pytest,
            env=self.config.env,
        )
        builder = LiveArchitectureGraphBuilder(shadow_config)
        graph = builder.build(persist=True)
        errors = graph.metrics.get("parse_errors", [])
        return (
            ProofResult(
                obligation_id="graph_rebuild",
                passed=not errors,
                status="passed" if not errors else "failed",
                evidence={"metrics": graph.metrics, "parse_errors": errors[:20]},
            ),
            graph.metrics,
        )

    def _architecture_quality_tests(self, manager: ShadowWorkspaceManager, shadow: ShadowRun) -> list[ProofResult]:
        root = Path(shadow.shadow_root)
        candidates = [root / "tests" / "test_architecture_quality.py", root / "tests" / "test_architecture_hardening.py"]
        results: list[ProofResult] = []
        for path in candidates:
            if not path.exists():
                continue
            rel = path.relative_to(root).as_posix()
            cmd = (python_executable(), "-m", "pytest", "-q", rel)
            result = manager.run_command(shadow, cmd, timeout=self.config.shadow_timeout)
            results.append(
                ProofResult(
                    obligation_id=f"architecture_quality:{rel}",
                    passed=result.exit_code == 0,
                    status="passed" if result.exit_code == 0 else "failed",
                    evidence={"exit_code": result.exit_code, "stdout": result.stdout, "stderr": result.stderr},
                )
            )
        if not results:
            results.append(ProofResult("architecture_quality", True, "not_available", {"reason": "no architecture quality test file found"}))
        return results

    def _relevant_tests(self, manager: ShadowWorkspaceManager, shadow: ShadowRun) -> list[ProofResult]:
        root = Path(shadow.shadow_root)
        test_paths: list[str] = []
        for rel in shadow.changed_files:
            stem = Path(rel).stem
            for candidate in (root / "tests").glob(f"test_*{stem}*.py") if (root / "tests").exists() else []:
                test_paths.append(candidate.relative_to(root).as_posix())
        test_paths = sorted(dict.fromkeys(test_paths))[:4]
        if not test_paths:
            return [ProofResult("relevant_tests", True, "not_available", {"reason": "no mapped tests discovered"})]
        result = manager.run_command(shadow, (python_executable(), "-m", "pytest", "-q", *test_paths), timeout=self.config.shadow_timeout)
        return [
            ProofResult(
                obligation_id="relevant_tests",
                passed=result.exit_code == 0,
                status="passed" if result.exit_code == 0 else "failed",
                evidence={"tests": test_paths, "exit_code": result.exit_code, "stdout": result.stdout, "stderr": result.stderr},
            )
        ]

    def _critical_tests(self, manager: ShadowWorkspaceManager, shadow: ShadowRun) -> list[ProofResult]:
        root = Path(shadow.shadow_root)
        candidates = [
            "tests/test_architecture_hardening.py",
            "tests/test_server_runtime_hardening.py",
            "tests/test_self_modification_sandbox_policy.py",
        ]
        existing = [path for path in candidates if (root / path).exists()]
        if not existing:
            return [ProofResult("critical_tests", True, "not_available", {"reason": "no critical subset found"})]
        result = manager.run_command(shadow, (python_executable(), "-m", "pytest", "-q", *existing), timeout=max(self.config.shadow_timeout, 45.0))
        return [
            ProofResult(
                obligation_id="critical_tests",
                passed=result.exit_code == 0,
                status="passed" if result.exit_code == 0 else "failed",
                evidence={"tests": existing, "exit_code": result.exit_code, "stdout": result.stdout, "stderr": result.stderr},
            )
        ]

    def _broader_pytest(self, manager: ShadowWorkspaceManager, shadow: ShadowRun) -> ProofResult:
        result = manager.run_command(shadow, (python_executable(), "-m", "pytest", "-q"), timeout=max(self.config.shadow_timeout, 120.0))
        return ProofResult(
            obligation_id="broader_pytest",
            passed=result.exit_code == 0,
            status="passed" if result.exit_code == 0 else "failed",
            evidence={"exit_code": result.exit_code, "stdout": result.stdout, "stderr": result.stderr},
        )

    def _safe_boot_check(self, manager: ShadowWorkspaceManager, shadow: ShadowRun) -> ProofResult:
        if not self.config.safe_boot_command:
            return ProofResult(
                obligation_id="safe_boot",
                passed=False,
                status="BOOT_HARNESS_UNAVAILABLE",
                evidence={"reason": "AURA_ASA_SAFE_BOOT_COMMAND is not configured"},
            )
        result = manager.run_command(shadow, self.config.safe_boot_command, timeout=self.config.shadow_timeout)
        return ProofResult(
            obligation_id="safe_boot",
            passed=result.exit_code == 0,
            status="passed" if result.exit_code == 0 else "failed",
            evidence={"command": list(self.config.safe_boot_command), "exit_code": result.exit_code, "stdout": result.stdout, "stderr": result.stderr},
        )

    def _minimal_runtime_simulation(self, manager: ShadowWorkspaceManager, shadow: ShadowRun) -> ProofResult:
        harness = Path(shadow.shadow_root) / "tests" / "live_harness_aura_v1.py"
        if not harness.exists():
            return ProofResult("minimal_runtime_simulation", True, "not_available", {"reason": "safe runtime harness not present"})
        result = manager.run_command(shadow, (python_executable(), str(harness.relative_to(Path(shadow.shadow_root)))), timeout=max(self.config.shadow_timeout, 60.0))
        return ProofResult(
            obligation_id="minimal_runtime_simulation",
            passed=result.exit_code == 0,
            status="passed" if result.exit_code == 0 else "failed",
            evidence={"exit_code": result.exit_code, "stdout": result.stdout, "stderr": result.stderr},
        )
