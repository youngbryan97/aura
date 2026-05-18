from __future__ import annotations

import ast
import logging
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from core.runtime.atomic_writer import atomic_write_text
from core.runtime.errors import record_degradation
from core.tasks.managed_command import ManagedCommandResult, run_project_command

logger = logging.getLogger("Aura.Verifier")
_CODE_VERIFIER_RECOVERABLE_ERRORS = (
    FileNotFoundError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)
CommandRunner = Callable[[tuple[str, ...], float], ManagedCommandResult]

# Dangerous AST node types that indicate potentially harmful code
_DANGEROUS_IMPORTS = frozenset({
    "subprocess", "shutil", "ctypes", "multiprocessing",
    "signal", "resource", "pty", "fcntl", "os", "sys", "importlib", "builtins",
})

_DANGEROUS_CALLS = frozenset({
    "exec", "eval", "compile", "__import__", "globals", "locals",
    "getattr", "setattr", "delattr", "open", "input", "breakpoint",
})

_BANNED_ATTRS = frozenset({
    "__class__", "__subclasses__", "__mro__", "__globals__", 
    "__subclasshook__", "__init__", "__func__", "__self__", "__dict__",
    "system", "popen", "execl", "execv", "execvp", "call", "run",
    "check_output", "check_call", "Popen"
})


@dataclass(frozen=True)
class ImportabilityReport:
    ok: bool
    syntax_ok: bool
    safety_ok: bool
    warnings: tuple[str, ...] = ()
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    error: str = ""


class CodeVerifier:
    """Implements Automated Program Repair (APR) safety patterns.
    Pattern: Generate -> Static Analysis -> Sandboxed Dry Run -> Deploy.

    SECURITY (v5.0): verify_importability avoids dynamic import execution
    (which would execute arbitrary top-level code). Instead it:
      1. Uses AST analysis to detect dangerous patterns
      2. Runs isolated bytecode compilation with timeout for import-shape checking
    """

    @staticmethod
    def verify_syntax(code: str) -> bool:
        """Stage 1: Static Analysis (AST parse)."""
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    @staticmethod
    def analyze_safety(code: str) -> dict:
        """Stage 1.5: AST-based safety analysis.
        Walks the AST to detect dangerous patterns without executing code.
        Returns {"safe": bool, "warnings": [str]}
        """
        warnings = []
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return {"safe": False, "warnings": ["Syntax error"]}

        for node in ast.walk(tree):
            # Check for dangerous imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    mod = alias.name.split(".")[0]
                    if mod in _DANGEROUS_IMPORTS:
                        warnings.append(f"Imports dangerous module: {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    mod = node.module.split(".")[0]
                    if mod in _DANGEROUS_IMPORTS:
                        warnings.append(f"Imports from dangerous module: {node.module}")
            # Check for dangerous function calls
            elif isinstance(node, ast.Call):
                func = node.func
                name = None
                if isinstance(func, ast.Name):
                    name = func.id
                elif isinstance(func, ast.Attribute):
                    name = func.attr
                if name and name in _DANGEROUS_CALLS:
                    warnings.append(f"Calls dangerous function: {name}()")
                if name and name in _BANNED_ATTRS:
                    warnings.append(f"Calls banned attribute/method: {name}")

            # Check for banned attribute access (M-04 / BUG-048)
            elif isinstance(node, ast.Attribute):
                if node.attr in _BANNED_ATTRS:
                    warnings.append(f"Forbidden attribute access: {node.attr}")
                
                # Check for os.path / os.system etc through attributes
                if isinstance(node.value, ast.Name) and node.value.id == "os":
                    if node.attr in ("system", "popen", "execl", "execv", "execvp"):
                        warnings.append(f"Calls os.{node.attr}()")

        return {"safe": len(warnings) == 0, "warnings": warnings}

    @staticmethod
    def verify_importability_report(
        code: str,
        module_name: str = "temp_module",
        timeout: int = 10,
        command_runner: CommandRunner | None = None,
    ) -> ImportabilityReport:
        """Stage 2: isolated compile dry run.

        The check writes code to a temporary module and runs ``py_compile`` in
        a bounded child process. It intentionally does not import or execute
        top-level code; runtime smoke tests belong in a governed sandbox.
        """
        syntax_ok = CodeVerifier.verify_syntax(code)
        if not syntax_ok:
            return ImportabilityReport(ok=False, syntax_ok=False, safety_ok=False, warnings=("Syntax error",))

        safety = CodeVerifier.analyze_safety(code)
        safety_ok = bool(safety["safe"])
        warnings = tuple(str(item) for item in safety["warnings"])
        if not safety_ok:
            logger.warning("Code failed safety analysis: %s", warnings)

        safe_name = CodeVerifier._safe_module_stem(module_name)
        try:
            with tempfile.TemporaryDirectory(prefix="aura_code_verify_") as tmpdir:
                tmp_path = Path(tmpdir) / f"{safe_name}.py"
                atomic_write_text(tmp_path, code, encoding="utf-8")
                runner = command_runner or (lambda command, limit: run_project_command(command, timeout_s=limit))
                result = runner((sys.executable, "-m", "py_compile", str(tmp_path)), float(timeout))

            if result.timed_out:
                logger.warning("Verification timed out after %ss", timeout)
                return ImportabilityReport(
                    ok=False,
                    syntax_ok=syntax_ok,
                    safety_ok=safety_ok,
                    warnings=warnings,
                    stdout=result.stdout,
                    stderr=result.stderr,
                    timed_out=True,
                )
            if not result.ok:
                stderr_sample = result.stderr[:500] if result.stderr else "No stderr"
                logger.warning("Verification compile failed: %s", stderr_sample)
                return ImportabilityReport(
                    ok=False,
                    syntax_ok=syntax_ok,
                    safety_ok=safety_ok,
                    warnings=warnings,
                    stdout=result.stdout,
                    stderr=result.stderr,
                )
            return ImportabilityReport(
                ok=safety_ok,
                syntax_ok=syntax_ok,
                safety_ok=safety_ok,
                warnings=warnings,
                stdout=result.stdout,
                stderr=result.stderr,
            )

        except _CODE_VERIFIER_RECOVERABLE_ERRORS as e:
            record_degradation('code_verifier', e)
            logger.error("Verifier internal error: %s", e)
            return ImportabilityReport(
                ok=False,
                syntax_ok=syntax_ok,
                safety_ok=safety_ok,
                warnings=warnings,
                error=str(e),
            )

    @staticmethod
    def verify_importability(
        code: str,
        module_name: str = "temp_module",
        timeout: int = 10,
        command_runner: CommandRunner | None = None,
    ) -> bool:
        return CodeVerifier.verify_importability_report(code, module_name, timeout, command_runner).ok

    @staticmethod
    def _safe_module_stem(module_name: str) -> str:
        stem = module_name.rsplit(".", 1)[-1]
        if stem.isidentifier():
            return stem
        return "candidate_module"
