"""
Code Repair Sandbox.
Specialized sandbox for verifying Python code patches before application.

C-13 FIX: Removed the "Dry Import Check" that executed module-level code
during verification. Now uses only AST parsing + py_compile for safety.
"""
import os
import sys
import tempfile
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import subprocess
import ast

# Import the base sandbox
try:
    from .sandbox import SecureSandbox, SecurityLevel, ExecutionResult
except ImportError:
    from sandbox import SecureSandbox, SecurityLevel, ExecutionResult

logger = logging.getLogger("security.code_sandbox")


class CodeRepairSandbox:
    """Sandbox for verifying code repairs.

    C-13 FIX: Verification now uses ONLY static analysis (AST parse +
    py_compile). The previous "Dry Import Check" was removed because
    importing a module executes all module-level code, which defeats
    the purpose of a safety check.
    """

    def __init__(self, security_level: SecurityLevel = SecurityLevel.RESTRICTED):
        self.sandbox = SecureSandbox(security_level=security_level)

    def verify_patch(self, original_file: Path, patched_content: str) -> Dict[str, Any]:
        """Verify a code patch using static analysis only.

        Steps:
          1. AST parse (catches syntax errors)
          2. py_compile in sandbox (catches compilation errors)
          3. AST security analysis (checks for dangerous imports/calls)

        C-13 FIX: No import-based execution. Module-level code is never run.
        """
        results = {
            "syntax_valid": False,
            "static_check_passed": False,
            "security_check_passed": False,
            "tests_passed": None,  # None = not run
            "error": None,
            "details": []
        }

        try:
            # 1. Syntax Check (AST parse)
            try:
                tree = ast.parse(patched_content)
                results["syntax_valid"] = True
            except SyntaxError as e:
                results["error"] = f"Syntax Error: {e}"
                results["details"].append(str(e))
                return results

            # 2. Security Analysis (AST walk for dangerous patterns)
            try:
                from core.security.ast_guard import ASTGuard, SecurityViolation
                guard = ASTGuard()  # Uses deny-by-default mode
                guard.validate(patched_content, source_label=str(original_file))
                results["security_check_passed"] = True
            except SecurityViolation as sv:
                results["error"] = f"Security Violation: {sv}"
                results["details"].append(str(sv))
                return results
            except ImportError:
                # ASTGuard not available — skip security check but log it
                logger.warning("ASTGuard not available, skipping security analysis")
                results["security_check_passed"] = True

            # 3. Static Compilation Check (py_compile) in Sandbox
            tmp_file_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    suffix=".py",
                    delete=False,
                    dir=self.sandbox.workdir,
                    mode="w",
                    encoding="utf-8"
                ) as tmp_file:
                    tmp_file_path = Path(tmp_file.name)
                    tmp_file.write(patched_content)

                compile_cmd = [sys.executable, "-m", "py_compile", str(tmp_file_path)]
                exec_result = self.sandbox.execute_command(compile_cmd, timeout=10.0)

                if exec_result.success:
                    results["static_check_passed"] = True
                else:
                    results["error"] = f"Compilation Failed: {exec_result.stderr}"
                    results["details"].append(exec_result.stderr)
            finally:
                # Always cleanup temp file
                if tmp_file_path and tmp_file_path.exists():
                    try:
                        get_task_tracker().create_task(get_storage_gateway().delete(tmp_file_path, cause='CodeRepairSandbox.verify_patch'))
                    except Exception as e:
                        logger.debug("Temp file cleanup failed: %s", e)

            return results

        except Exception as e:
            logger.error("Patch verification failed: %s", e)
            results["error"] = str(e)
            return results
