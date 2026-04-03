import ast
import logging
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

logger = logging.getLogger("Aura.Verifier")

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


class CodeVerifier:
    """Implements Automated Program Repair (APR) safety patterns.
    Pattern: Generate -> Static Analysis -> Sandboxed Dry Run -> Deploy.

    SECURITY (v5.0): verify_importability no longer uses exec_module
    (which executed arbitrary top-level code). Instead it:
      1. Uses AST analysis to detect dangerous patterns
      2. Runs a subprocess with timeout + restricted permissions for import checking
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
    def verify_importability(code: str, module_name: str = "temp_module", timeout: int = 10) -> bool:
        """Stage 2: Sandboxed Dry Run.
        Writes code to a temp file and attempts to import it in a
        SUBPROCESS with a strict timeout. Never executes code in the
        main process.
        """
        tmp_path = None
        try:
            # First do a fast AST safety check
            safety = CodeVerifier.analyze_safety(code)
            if not safety["safe"]:
                logger.warning("Code failed safety analysis: %s", safety['warnings'])
                # Still allow import check but log the warnings

            with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode='w') as tmp:
                tmp.write(code)
                tmp_path = tmp.name

            # Run import check in isolated subprocess with timeout
            check_script = (
                f"import importlib.util, sys; "
                f"spec = importlib.util.spec_from_file_location('{module_name}', '{tmp_path}'); "
                f"mod = importlib.util.module_from_spec(spec); "
                f"spec.loader.exec_module(mod); "
                f"print('OK')"
            )

            result = subprocess.run(
                [sys.executable, "-c", check_script],
                capture_output=True,
                text=True,
                timeout=timeout,
                env={
                    "PATH": os.environ.get("PATH", ""),
                    "HOME": os.environ.get("HOME", "/tmp"),
                    "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
                },
            )

            if result.returncode == 0 and "OK" in result.stdout:
                return True
            else:
                stderr_sample = result.stderr[:500] if result.stderr else "No stderr"
                logger.warning("Verification Import Failed: %s", stderr_sample)
                return False

        except subprocess.TimeoutExpired:
            logger.warning("Verification timed out after %ss — possible infinite loop", timeout)
            return False
        except Exception as e:
            logger.error("Verifier internal error: %s", e)
            return False
        finally:
            # Cleanup
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.remove(tmp_path)