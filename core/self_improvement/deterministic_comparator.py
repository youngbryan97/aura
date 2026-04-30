"""core/self_improvement/deterministic_comparator.py — Behavioral comparison engine.

Runs candidate code in sandbox, compares outputs against expected behavior.
This is the paper's cell-by-cell comparison step adapted for Aura modules:
instead of table cells we compare test verdicts, output values, and latency.

Uses deterministic grading — no LLM judge. Authority is tests + sandbox.
"""
from __future__ import annotations

import ast
import logging
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.self_improvement.interface_contract import (
    CandidateModule,
    ComparisonReport,
    ModuleSpec,
    TestVerdict,
)
from core.self_improvement.blinded_workspace import BlindedWorkspace

logger = logging.getLogger("Aura.DeterministicComparator")


class DeterministicComparator:
    """Runs candidate implementations against spec tests and grades deterministically."""

    def __init__(self, timeout: int = 30):
        self.timeout = timeout

    async def compare(
        self,
        candidate: CandidateModule,
        spec: ModuleSpec,
        workspace: BlindedWorkspace,
    ) -> ComparisonReport:
        """Run full behavioral comparison.

        1. Validate syntax
        2. Check imports resolve
        3. Verify public surface is preserved
        4. Run tests in sandbox
        5. Aggregate results
        """
        report = ComparisonReport()
        start = time.monotonic()

        # 1. Syntax validation
        try:
            tree = ast.parse(candidate.source_code)
            report.syntax_valid = True
        except SyntaxError as e:
            report.syntax_valid = False
            report.verdicts.append(TestVerdict(
                test_name="__syntax__", passed=False,
                error_message=f"SyntaxError: {e}",
            ))
            report.generated_at = time.time()
            return report

        # 2. Public surface check
        report.public_surface_preserved = self._check_public_surface(tree, spec)
        if not report.public_surface_preserved:
            report.verdicts.append(TestVerdict(
                test_name="__public_surface__", passed=False,
                error_message="Public surface not preserved — missing functions/classes",
            ))

        # 3. Import validation (static check)
        report.imports_valid = self._check_imports(tree, spec)

        # 4. Write candidate to workspace and run tests
        candidate_path = workspace.workspace_dir / spec.module_path
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        candidate_path.write_text(candidate.source_code, encoding="utf-8")

        # 5. Run tests
        test_verdicts = self._run_tests(spec, workspace)
        report.verdicts.extend(test_verdicts)

        # 6. Aggregate
        test_only = [v for v in report.verdicts if not v.test_name.startswith("__")]
        report.total_tests = len(test_only)
        report.passed_tests = sum(1 for v in test_only if v.passed)
        report.failed_tests = sum(1 for v in test_only if not v.passed)
        report.aggregate_pass_rate = (
            report.passed_tests / report.total_tests if report.total_tests > 0 else 0.0
        )

        # Metrics for promotion gate
        report.metrics = {
            "test_pass_rate": report.aggregate_pass_rate,
            "syntax_valid": 1.0 if report.syntax_valid else 0.0,
            "public_surface": 1.0 if report.public_surface_preserved else 0.0,
            "imports_valid": 1.0 if report.imports_valid else 0.0,
            "total_tests": float(report.total_tests),
        }

        elapsed = time.monotonic() - start
        logger.info(
            "Comparison complete for %s: %d/%d passed (%.1f%%) in %.2fs",
            spec.module_path, report.passed_tests, report.total_tests,
            report.aggregate_pass_rate * 100, elapsed,
        )
        return report

    def _check_public_surface(self, tree: ast.Module, spec: ModuleSpec) -> bool:
        """Verify all public names from spec exist in candidate."""
        candidate_names = set()
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                candidate_names.add(node.name)
            elif isinstance(node, ast.ClassDef):
                candidate_names.add(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        candidate_names.add(target.id)

        required = spec.interface.public_names
        if not required:
            return True
        missing = required - candidate_names
        if missing:
            logger.warning("Missing public names in candidate: %s", missing)
            return False
        return True

    def _check_imports(self, tree: ast.Module, spec: ModuleSpec) -> bool:
        """Basic static check that imports look reasonable."""
        # We check that the candidate doesn't import forbidden modules
        forbidden = {"subprocess", "socket", "urllib", "requests", "http.client"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in forbidden:
                        return False
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.split(".")[0] in forbidden:
                    return False
        return True

    def _run_tests(self, spec: ModuleSpec, workspace: BlindedWorkspace) -> List[TestVerdict]:
        """Run the spec's test cases in an isolated subprocess."""
        verdicts: List[TestVerdict] = []

        if not spec.test_cases:
            return verdicts

        # Build a minimal test runner script
        test_files = set()
        for tc in spec.test_cases:
            if tc.file_path:
                test_path = workspace.test_dir / Path(tc.file_path).name
                if test_path.exists():
                    test_files.add(str(test_path))

        if not test_files:
            return verdicts

        # Run pytest in subprocess with the workspace on sys.path
        for test_file in test_files:
            try:
                result = subprocess.run(
                    [
                        sys.executable, "-m", "pytest", test_file,
                        "-v", "--tb=short", "--no-header", "-q",
                        f"--rootdir={workspace.workspace_dir}",
                    ],
                    capture_output=True, text=True,
                    timeout=self.timeout,
                    cwd=str(workspace.workspace_dir),
                    env={
                        **dict(__import__("os").environ),
                        "PYTHONPATH": str(workspace.workspace_dir),
                        "PYTHONDONTWRITEBYTECODE": "1",
                    },
                )

                # Parse pytest output for individual test results
                verdicts.extend(
                    self._parse_pytest_output(result.stdout, result.stderr, result.returncode)
                )
            except subprocess.TimeoutExpired:
                verdicts.append(TestVerdict(
                    test_name=f"__timeout__{Path(test_file).name}",
                    passed=False, error_message=f"Test timed out after {self.timeout}s",
                ))
            except Exception as e:
                verdicts.append(TestVerdict(
                    test_name=f"__error__{Path(test_file).name}",
                    passed=False, error_message=str(e),
                ))

        return verdicts

    def _parse_pytest_output(
        self, stdout: str, stderr: str, returncode: int
    ) -> List[TestVerdict]:
        """Parse pytest verbose output into TestVerdicts."""
        verdicts: List[TestVerdict] = []
        for line in stdout.splitlines():
            line = line.strip()
            if " PASSED" in line:
                name = line.split(" PASSED")[0].strip()
                name = name.split("::")[-1] if "::" in name else name
                verdicts.append(TestVerdict(test_name=name, passed=True))
            elif " FAILED" in line:
                name = line.split(" FAILED")[0].strip()
                name = name.split("::")[-1] if "::" in name else name
                verdicts.append(TestVerdict(
                    test_name=name, passed=False,
                    error_message=stderr[:500] if stderr else "Test failed",
                    stderr=stderr[:1000] if stderr else "",
                ))
            elif " ERROR" in line:
                name = line.split(" ERROR")[0].strip()
                name = name.split("::")[-1] if "::" in name else name
                verdicts.append(TestVerdict(
                    test_name=name, passed=False,
                    error_message=f"Collection/setup error: {stderr[:500]}",
                ))

        # If no individual results parsed but returncode != 0
        if not verdicts and returncode != 0:
            verdicts.append(TestVerdict(
                test_name="__pytest_overall__", passed=False,
                error_message=f"pytest exited with code {returncode}",
                stderr=stderr[:1000] if stderr else "",
                stdout=stdout[:1000] if stdout else "",
            ))

        return verdicts


__all__ = ["DeterministicComparator"]
