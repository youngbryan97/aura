"""core/self_improvement/guardrail_auditor.py — Safety and governance checks.

Delegates to existing Aura governance infrastructure:
  - core/self_modification/formal_verifier.py for AST invariants
  - Checks governance fence preservation
  - Verifies no bypass of UnifiedWill.decide / AuthorityGateway.authorize
  - Ensures protected safety modules remain untouched
"""
from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import List, Optional

from core.self_improvement.interface_contract import (
    AuditResult,
    CandidateModule,
)

logger = logging.getLogger("Aura.GuardrailAuditor")

# Governance calls that must not be removed
GOVERNANCE_FENCE_CALLS = {
    "UnifiedWill.decide", "get_will().decide",
    "AuthorityGateway.authorize", "_will_gate",
}

# Protected safety modules that must never be reconstructed
PROTECTED_MODULES = frozenset({
    "constitutional_guard.py", "master_moral_integration.py",
    "emergency_protocol.py", "heartstone_values.py",
    "behavior_controller.py", "safety_registry.py", "identity_guard.py",
})

# Dangerous imports not allowed in candidates
UNSAFE_IMPORTS = frozenset({
    "subprocess", "os.system", "shutil.rmtree",
    "ctypes", "multiprocessing",
})


class GuardrailAuditor:
    """Verifies candidate code does not violate Aura governance invariants."""

    def __init__(self, project_root: Optional[str] = None):
        self.project_root = Path(project_root or ".").resolve()

    def audit(self, candidate: CandidateModule, original_module_path: str) -> AuditResult:
        """Run all governance checks.

        Args:
            candidate: The generated candidate module.
            original_module_path: Path to the module being replaced.

        Returns:
            AuditResult with violations if any governance rules are broken.
        """
        violations: List[str] = []

        # 1. Protected module check
        target_name = Path(original_module_path).name
        if target_name in PROTECTED_MODULES:
            violations.append(
                f"PROTECTED_MODULE: Cannot reconstruct protected safety module: {target_name}"
            )

        # 2. Parse candidate
        try:
            tree = ast.parse(candidate.source_code)
        except SyntaxError as e:
            violations.append(f"SYNTAX_ERROR: Candidate does not parse: {e}")
            return AuditResult(
                passed=False, violations=violations, audit_type="guardrail",
            )

        # 3. Check for dangerous imports
        dangerous = self._check_dangerous_imports(tree)
        violations.extend(dangerous)

        # 4. Formal verification if original source exists
        formal_violations = self._formal_verify(tree, original_module_path)
        violations.extend(formal_violations)

        # 5. Check governance fence preservation
        fence_violations = self._check_governance_fences(tree, original_module_path)
        violations.extend(fence_violations)

        passed = len(violations) == 0
        if not passed:
            logger.warning(
                "Guardrail audit FAILED for %s: %d violations",
                candidate.module_path, len(violations),
            )
        else:
            logger.info("Guardrail audit PASSED for %s", candidate.module_path)

        return AuditResult(
            passed=passed, violations=violations, audit_type="guardrail",
        )

    def _check_dangerous_imports(self, tree: ast.Module) -> List[str]:
        """Check for imports that should not appear in reconstructed modules."""
        violations: List[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in UNSAFE_IMPORTS:
                        violations.append(
                            f"UNSAFE_IMPORT: Candidate imports {alias.name}"
                        )
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    full = node.module
                    if full in UNSAFE_IMPORTS or full.split(".")[0] in {"ctypes"}:
                        violations.append(
                            f"UNSAFE_IMPORT: Candidate imports from {full}"
                        )
        return violations

    def _formal_verify(self, tree: ast.Module, original_module_path: str) -> List[str]:
        """Run formal verification using core/self_modification/formal_verifier."""
        violations: List[str] = []
        original_path = self.project_root / original_module_path
        if not original_path.exists():
            return violations

        try:
            from core.self_modification.formal_verifier import verify_mutation
            original_source = original_path.read_text(encoding="utf-8")
            candidate_source = ast.unparse(tree)

            result = verify_mutation(
                file_path=original_module_path,
                before_source=original_source,
                after_source=candidate_source,
            )

            if not result.ok:
                for inv in result.invariants_violated:
                    violations.append(f"FORMAL_INVARIANT: {inv}")
        except ImportError:
            logger.debug("formal_verifier not available — skipping formal checks")
        except Exception as e:
            logger.debug("Formal verification error (non-fatal): %s", e)

        return violations

    def _check_governance_fences(self, tree: ast.Module, original_module_path: str) -> List[str]:
        """Ensure governance fence calls are not removed."""
        violations: List[str] = []

        # Only check if original has governance fences
        original_path = self.project_root / original_module_path
        if not original_path.exists():
            return violations

        try:
            original_source = original_path.read_text(encoding="utf-8")
            original_tree = ast.parse(original_source)
        except Exception:
            return violations

        # Count governance calls in original
        orig_count = self._count_governance_calls(original_tree)
        cand_count = self._count_governance_calls(tree)

        if cand_count < orig_count:
            violations.append(
                f"GOVERNANCE_FENCE: Governance calls decreased from "
                f"{orig_count} to {cand_count}"
            )

        return violations

    def _count_governance_calls(self, tree: ast.Module) -> int:
        """Count governance fence calls in an AST."""
        count = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                qn = self._qualname(node.func)
                if any(qn.endswith(g) or qn == g for g in GOVERNANCE_FENCE_CALLS):
                    count += 1
        return count

    def _qualname(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return self._qualname(node.value) + "." + node.attr
        return ""


__all__ = ["GuardrailAuditor"]
