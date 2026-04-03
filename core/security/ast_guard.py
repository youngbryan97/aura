"""AST Guard: Code Safety Verification
-----------------------------------
Analyzes Python code abstract syntax trees (AST) to detect potentially
unsafe operations before execution. Essential for running LLM-generated code.

M-04 FIX: Default mode is now deny-all imports (secure by default).
When no allowed_modules are specified, ALL imports are blocked.
"""

import ast
import logging
from typing import List, Optional, Set

logger = logging.getLogger("Aura.Security")

# Safe modules that are always allowed for basic operations
DEFAULT_SAFE_MODULES = frozenset({
    "math", "statistics", "datetime", "collections", "itertools",
    "functools", "operator", "string", "re", "json", "enum",
    "dataclasses", "typing", "abc", "copy", "pprint",
    "decimal", "fractions", "random", "textwrap",
    "pathlib",  # Read-only path ops
})

# Modules that must NEVER be allowed
FORBIDDEN_MODULES = frozenset({
    "os", "subprocess", "shutil", "sys", "ctypes", "socket",
    "http", "urllib", "requests", "httpx", "aiohttp",
    "multiprocessing", "signal", "importlib", "builtins",
    "code", "codeop", "compileall", "runpy",
    "pickle", "shelve", "marshal",
    "webbrowser", "antigravity",
    "pty", "pdb", "trace", "gc",
})


class SecurityViolation(Exception):
    pass


class ASTGuard(ast.NodeVisitor):
    """AST-based code safety guard.

    M-04 FIX: Default mode is DENY-ALL imports. You must explicitly
    specify allowed_modules to permit any imports. The DEFAULT_SAFE_MODULES
    constant provides a reasonable starting set for computation-only code.
    """

    def __init__(
        self,
        allowed_modules: Optional[List[str]] = None,
        unsafe_builtins: Optional[List[str]] = None,
        deny_all_imports: bool = True,
    ):
        # M-04 FIX: Default to safe modules if none specified and deny_all is True
        if allowed_modules is not None:
            self.allowed_modules: Set[str] = set(allowed_modules)
        elif deny_all_imports:
            self.allowed_modules = set(DEFAULT_SAFE_MODULES)
        else:
            self.allowed_modules = set()  # Empty = allow all (legacy permissive)

        self.deny_all_imports = deny_all_imports
        self.unsafe_builtins = set(
            unsafe_builtins
            or ["eval", "exec", "compile", "__import__", "globals", "locals",
                "getattr", "setattr", "delattr", "open", "input", "breakpoint"]
        )
        self.violations: List[str] = []

    def visit_Import(self, node):
        for alias in node.names:
            self._check_import(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module:
            self._check_import(node.module)
        self.generic_visit(node)

    def visit_Attribute(self, node):
        # Block introspection escapes
        if node.attr in ("__class__", "__subclasses__", "__mro__", "__globals__", 
                         "__subclasshook__", "__init__", "__func__", "__self__", "__dict__"):
            self.violations.append(f"Forbidden attribute access: {node.attr}")
        self.generic_visit(node)

    def visit_Call(self, node):
        # Check for calls to unsafe builtins
        if isinstance(node.func, ast.Name):
            if node.func.id in self.unsafe_builtins:
                self.violations.append(f"Call to unsafe builtin: {node.func.id}")
        # Check for attribute calls like os.system() or obj.__subclasses__()
        elif isinstance(node.func, ast.Attribute):
            if node.func.attr in ("system", "popen", "exec", "call", "run",
                                   "check_output", "check_call", "Popen",
                                   "spawnl", "spawnv", "execv", "execve",
                                   "__subclasses__", "globals"):
                self.violations.append(
                    f"Call to dangerous method: {node.func.attr}"
                )
        self.generic_visit(node)

    def _check_import(self, module_name: str):
        base_module = module_name.split(".")[0]

        # Always block forbidden modules
        if base_module in FORBIDDEN_MODULES:
            self.violations.append(f"Import of forbidden module: {module_name}")
            return

        # If we have an allow list, check against it
        if self.allowed_modules and base_module not in self.allowed_modules:
            self.violations.append(f"Import of restricted module: {module_name}")

    def validate(self, code_str: str, source_label="<string>"):
        """Parses and validates the code string.
        Raises SecurityViolation if unsafe patterns are found.
        """
        try:
            tree = ast.parse(code_str, filename=source_label)
        except SyntaxError as e:
            raise SecurityViolation(f"Syntax Error: {e}")

        self.violations = []
        self.visit(tree)

        if self.violations:
            violation_msg = "; ".join(self.violations)
            logger.warning("Security blocked code execution: %s", violation_msg)
            raise SecurityViolation(
                f"Code violated security policy: {violation_msg}"
            )

        return True