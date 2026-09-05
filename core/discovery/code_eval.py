"""SafeCodeEvaluator — AST-allowlisted Python sandbox for discovery.

Layered defence:
  1. AST allowlist — only the node types / call names in the lists
     below survive parse-time.  Any import, attribute mutation, dunder
     access, exec/eval, or unknown call name fails *before* code runs.
  2. Subprocess isolation — the candidate runs through F4's
     ``SafeMutationEvaluator`` so it inherits rlimits, timeouts,
     stdout/stderr capture, and quarantine on failure.
  3. Test execution — the candidate must define a function whose name
     the caller specifies; tests are tuples of ``(args, expected)`` and
     are evaluated by passing args through ``*args``.

Returns a ``DiscoveryEvaluation`` with a typed outcome (passed,
ast_violation, compile_fail, runtime, assertion, timeout, oom).  An
F4 quarantine entry exists for every non-passed outcome.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.self_modification.mutation_safety import (
    MutationOutcome,
    QuarantineStore,
    SafeMutationEvaluator,
)


# Python AST nodes that are safe inside a function-body candidate.
ALLOWED_AST = {
    ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return,
    ast.Assign, ast.AugAssign, ast.For, ast.While, ast.If, ast.Compare,
    ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Call, ast.Name, ast.Load,
    ast.Store, ast.Constant, ast.List, ast.Tuple, ast.Subscript,
    ast.Slice, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv,
    ast.Mod, ast.Pow, ast.USub, ast.UAdd, ast.Eq, ast.NotEq, ast.Lt,
    ast.LtE, ast.Gt, ast.GtE, ast.And, ast.Or, ast.Not, ast.Break,
    ast.Continue, ast.Pass, ast.Expr, ast.Assert, ast.IfExp,
}

# Built-in function names the candidate may call.
ALLOWED_CALLS = frozenset(
    {
        "range", "len", "min", "max", "sum", "abs", "sorted",
        "int", "float", "str", "list", "tuple", "enumerate",
        "all", "any", "map", "filter", "reversed",
    }
)


class ASTViolation(ValueError):
    """Raised when a candidate fails the AST allowlist."""


@dataclass
class DiscoveryEvaluation:
    outcome: str  # "passed" | "ast_violation" | "compile_fail" | "runtime" |
                  # "assertion" | "timeout" | "oom"
    passed: int = 0
    total: int = 0
    error: Optional[str] = None
    quarantine_path: Optional[str] = None
    stdout: str = ""
    stderr: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.outcome == "passed" and self.total > 0 and self.passed == self.total

    def to_dict(self) -> Dict[str, Any]:
        return {
            "outcome": self.outcome,
            "passed": self.passed,
            "total": self.total,
            "error": self.error,
            "quarantine_path": self.quarantine_path,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "metadata": dict(self.metadata),
        }


def _audit_ast(code: str) -> None:
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if type(node) not in ALLOWED_AST:
            raise ASTViolation(f"disallowed node: {type(node).__name__}")
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ASTViolation("imports are not allowed")
        if isinstance(node, ast.Attribute):
            # No attribute access at all — closes off __subclasses__,
            # __mro__, and similar dunder gadgets.
            raise ASTViolation("attribute access is not allowed")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ASTViolation("only direct function-name calls are allowed")
            if node.func.id not in ALLOWED_CALLS:
                raise ASTViolation(f"disallowed call: {node.func.id}")
        if isinstance(node, ast.Name):
            if node.id.startswith("__") and node.id.endswith("__"):
                raise ASTViolation(f"disallowed dunder identifier: {node.id}")


def _build_runner(fn_name: str, tests: Sequence[Tuple[Any, ...]]) -> str:
    """Wrap the candidate so the subprocess loads it, runs every test
    case, and asserts on the expected outcome.  Errors propagate as
    AssertionError or RuntimeError into F4's typed evaluator."""
    payload = repr([(list(args), expected) for args, expected in tests])
    return (
        f"\n# --- runner injected by SafeCodeEvaluator ---\n"
        f"_TESTS = {payload}\n"
        f"_PASSED = 0\n"
        f"for _args, _expected in _TESTS:\n"
        f"    _got = {fn_name}(*_args)\n"
        f"    assert _got == _expected, "
        f"f'expected={{_expected!r}} got={{_got!r}} on args={{_args!r}}'\n"
        f"    _PASSED += 1\n"
        f"assert _PASSED == len(_TESTS), 'not all tests passed'\n"
    )


class SafeCodeEvaluator:
    def __init__(
        self,
        *,
        timeout_seconds: float = 5.0,
        memory_mb: int = 256,
        quarantine: Optional[QuarantineStore] = None,
    ):
        self.timeout_seconds = float(timeout_seconds)
        self.memory_mb = int(memory_mb)
        self.quarantine = quarantine or QuarantineStore()
        self._delegate = SafeMutationEvaluator(
            timeout_seconds=self.timeout_seconds,
            memory_mb=self.memory_mb,
            quarantine=self.quarantine,
        )

    def evaluate(
        self,
        code: str,
        fn_name: str,
        tests: Sequence[Tuple[Tuple[Any, ...], Any]],
    ) -> DiscoveryEvaluation:
        if not fn_name.isidentifier():
            return DiscoveryEvaluation(
                outcome="ast_violation",
                error=f"invalid fn_name: {fn_name!r}",
                metadata={"reason": "fn_name_not_identifier"},
            )
        try:
            _audit_ast(code)
        except ASTViolation as exc:
            return DiscoveryEvaluation(
                outcome="ast_violation",
                error=str(exc),
                total=len(tests),
            )
        except SyntaxError as exc:
            return DiscoveryEvaluation(
                outcome="compile_fail",
                error=str(exc),
                total=len(tests),
            )

        runner = _build_runner(fn_name, tests)
        full_source = code + "\n" + runner

        diag = self._delegate.evaluate(full_source)
        outcome_map = {
            MutationOutcome.PASSED: ("passed", len(tests)),
            MutationOutcome.COMPILE_FAIL: ("compile_fail", 0),
            MutationOutcome.IMPORT_FAIL: ("ast_violation", 0),
            MutationOutcome.RUNTIME_EXCEPTION: ("runtime", 0),
            MutationOutcome.ASSERTION_FAIL: ("assertion", 0),
            MutationOutcome.TIMEOUT: ("timeout", 0),
            MutationOutcome.OOM: ("oom", 0),
        }
        outcome_str, passed = outcome_map.get(diag.outcome, ("runtime", 0))
        return DiscoveryEvaluation(
            outcome=outcome_str,
            passed=passed,
            total=len(tests),
            error=None if outcome_str == "passed" else diag.traceback_text or None,
            quarantine_path=diag.quarantine_path,
            stdout=diag.stdout[-2000:],
            stderr=diag.stderr[-2000:],
            metadata={
                "runtime_seconds": diag.runtime_seconds,
                "exit_code": diag.exit_code,
            },
        )
