"""core/self_improvement/hardcoding_auditor.py — Anti-cheating detection.

The paper is obsessed with preventing agents from cheating: agents cannot
see original results, original code, or forbidden paths, and the system
audits for hardcoded statistical outputs.

This auditor does the same for Aura:
  1. Scans candidate code for hardcoded values matching test expectations
  2. Detects forbidden path access via the workspace access log
  3. Checks for eval/exec of test data
  4. Detects constant-return functions that should compute
"""
from __future__ import annotations

import ast
import logging
import re
from typing import Any, Dict, List, Set

from core.self_improvement.interface_contract import (
    AuditResult,
    CandidateModule,
    ModuleSpec,
)
from core.self_improvement.blinded_workspace import BlindedWorkspace

logger = logging.getLogger("Aura.HardcodingAuditor")


class HardcodingAuditor:
    """Detects cheating in candidate implementations."""

    def audit(
        self,
        candidate: CandidateModule,
        spec: ModuleSpec,
        workspace: BlindedWorkspace,
    ) -> AuditResult:
        """Run all anti-cheating checks.

        Returns an AuditResult. passed=False if any violation is detected.
        """
        violations: List[str] = []

        # 1. Check for forbidden path access
        forbidden_violations = self._check_forbidden_access(workspace)
        violations.extend(forbidden_violations)

        # 2. Parse candidate AST
        try:
            tree = ast.parse(candidate.source_code)
        except SyntaxError:
            # Can't audit unparseable code — but this is caught by comparator
            return AuditResult(
                passed=len(violations) == 0,
                violations=violations,
                audit_type="hardcoding",
            )

        # 3. Check for hardcoded magic values
        hardcoded = self._check_hardcoded_values(tree, spec)
        violations.extend(hardcoded)

        # 4. Check for eval/exec of external data
        eval_violations = self._check_eval_exec(tree)
        violations.extend(eval_violations)

        # 5. Check for constant-return functions
        const_violations = self._check_constant_returns(tree, spec)
        violations.extend(const_violations)

        # 6. Check for file reads of forbidden paths
        file_read_violations = self._check_file_reads(tree, workspace)
        violations.extend(file_read_violations)

        passed = len(violations) == 0
        if not passed:
            logger.warning(
                "Hardcoding audit FAILED for %s: %d violations",
                candidate.module_path, len(violations),
            )
        else:
            logger.info("Hardcoding audit PASSED for %s", candidate.module_path)

        return AuditResult(
            passed=passed,
            violations=violations,
            audit_type="hardcoding",
        )

    def _check_forbidden_access(self, workspace: BlindedWorkspace) -> List[str]:
        """Check workspace access log for forbidden path access."""
        violations: List[str] = []
        for path in workspace.access_log:
            if workspace.is_forbidden(path):
                violations.append(f"FORBIDDEN_ACCESS: Accessed blocked path: {path}")
        return violations

    def _check_hardcoded_values(self, tree: ast.Module, spec: ModuleSpec) -> List[str]:
        """Detect suspicious hardcoded numeric/string constants.

        Looks for functions that return literal values matching known
        test expectations — a sign of memorized answers.
        """
        violations: List[str] = []

        # Collect all return-value constants
        for node in ast.walk(tree):
            if isinstance(node, ast.Return) and node.value:
                if isinstance(node.value, ast.Constant):
                    val = node.value.value
                    # Flag suspiciously specific floats (likely hardcoded stats)
                    if isinstance(val, float) and val != 0.0 and val != 1.0:
                        # Check if this is unreasonably specific
                        s = str(val)
                        if len(s.split(".")[-1]) > 4:
                            violations.append(
                                f"HARDCODED_RETURN: Suspiciously specific constant return "
                                f"value {val} (line ~{getattr(node, 'lineno', '?')})"
                            )
        return violations

    def _check_eval_exec(self, tree: ast.Module) -> List[str]:
        """Detect eval() or exec() calls that could execute test data."""
        violations: List[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ("eval", "exec", "compile"):
                        violations.append(
                            f"EVAL_EXEC: Call to {node.func.id}() detected "
                            f"(line ~{getattr(node, 'lineno', '?')})"
                        )
        return violations

    def _check_constant_returns(self, tree: ast.Module, spec: ModuleSpec) -> List[str]:
        """Detect functions that should compute but always return a constant."""
        violations: List[str] = []

        # Get names of functions that should have logic
        expected_compute_fns: Set[str] = set()
        for func in spec.interface.functions:
            if len(func.parameters) > 0:  # Functions with params should compute
                expected_compute_fns.add(func.name)
        for cls in spec.interface.classes:
            for method in cls.methods:
                if method.name != "__init__" and len(method.parameters) > 1:
                    expected_compute_fns.add(f"{cls.name}.{method.name}")

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name in expected_compute_fns:
                    # Check if function body is just `return <constant>`
                    body_stmts = [s for s in node.body
                                  if not isinstance(s, ast.Expr) or
                                  not isinstance(s.value, ast.Constant)]  # skip docstrings
                    if len(body_stmts) == 1:
                        stmt = body_stmts[0]
                        if (isinstance(stmt, ast.Return) and stmt.value and
                                isinstance(stmt.value, ast.Constant)):
                            violations.append(
                                f"CONSTANT_RETURN: Function {node.name} returns a constant "
                                f"but should compute from inputs "
                                f"(line ~{getattr(node, 'lineno', '?')})"
                            )
        return violations

    def _check_file_reads(self, tree: ast.Module, workspace: BlindedWorkspace) -> List[str]:
        """Detect attempts to read files outside the workspace."""
        violations: List[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # Detect open() calls
                func = node.func
                if isinstance(func, ast.Name) and func.id == "open":
                    # Check if the path argument contains forbidden paths
                    if node.args:
                        arg = node.args[0]
                        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                            if workspace.is_forbidden(arg.value):
                                violations.append(
                                    f"FORBIDDEN_FILE_READ: Attempt to read forbidden "
                                    f"file: {arg.value}"
                                )
                # Detect Path.read_text() etc.
                elif isinstance(func, ast.Attribute):
                    if func.attr in ("read_text", "read_bytes", "open"):
                        if isinstance(func.value, ast.Call):
                            # Path(...).read_text()
                            if (isinstance(func.value.func, ast.Name) and
                                    func.value.func.id == "Path"):
                                if func.value.args:
                                    arg = func.value.args[0]
                                    if (isinstance(arg, ast.Constant) and
                                            isinstance(arg.value, str)):
                                        if workspace.is_forbidden(arg.value):
                                            violations.append(
                                                f"FORBIDDEN_FILE_READ: Path access to "
                                                f"forbidden file: {arg.value}"
                                            )
        return violations


__all__ = ["HardcodingAuditor"]
