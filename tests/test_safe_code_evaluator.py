"""Tests for SafeCodeEvaluator.

The evaluator is two-layered: AST allowlist + F4 subprocess
isolation.  Tests target both layers individually plus the typed
outcome surface.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from core.discovery.code_eval import (
    ALLOWED_CALLS,
    ASTViolation,
    DiscoveryEvaluation,
    SafeCodeEvaluator,
    _audit_ast,
)
from core.self_modification.mutation_safety import QuarantineStore


# ---------------------------------------------------------------------------
# AST allowlist (pre-subprocess)
# ---------------------------------------------------------------------------
def test_audit_ast_accepts_simple_function():
    _audit_ast("def f(a, b):\n    return a + b\n")


def test_audit_ast_rejects_import():
    with pytest.raises(ASTViolation):
        _audit_ast("import os\ndef f(): return 1\n")


def test_audit_ast_rejects_from_import():
    with pytest.raises(ASTViolation):
        _audit_ast("from os import system\ndef f(): return 1\n")


def test_audit_ast_rejects_attribute_access():
    with pytest.raises(ASTViolation):
        _audit_ast("def f():\n    return ().__class__\n")


def test_audit_ast_rejects_unknown_call():
    with pytest.raises(ASTViolation):
        _audit_ast("def f():\n    return open('x')\n")


def test_audit_ast_rejects_dunder_name():
    with pytest.raises(ASTViolation):
        _audit_ast("def f():\n    __builtins__.exec('1')\n")


def test_audit_ast_rejects_call_via_subscript():
    """Indirect calls like x[0]() must fail because the call's func is
    not a direct ast.Name."""
    with pytest.raises(ASTViolation):
        _audit_ast("def f(x):\n    return x[0]()\n")


def test_audit_ast_accepts_allowed_calls():
    code = "def f(xs):\n    return sum(sorted(xs))\n"
    _audit_ast(code)  # sum + sorted are in the allowlist


# ---------------------------------------------------------------------------
# SafeCodeEvaluator end-to-end
# ---------------------------------------------------------------------------
@pytest.fixture
def evaluator(tmp_path: Path) -> SafeCodeEvaluator:
    return SafeCodeEvaluator(
        timeout_seconds=5.0,
        memory_mb=256,
        quarantine=QuarantineStore(tmp_path / "quarantine"),
    )


def test_passing_candidate(evaluator: SafeCodeEvaluator):
    code = "def add(a, b):\n    return a + b\n"
    tests = [((1, 1), 2), ((3, 4), 7), ((-5, 10), 5)]
    result = evaluator.evaluate(code, "add", tests)
    assert result.outcome == "passed"
    assert result.ok is True
    assert result.passed == 3
    assert result.total == 3
    assert result.quarantine_path is None


def test_failing_assertion_returns_assertion_outcome(evaluator):
    code = "def add(a, b):\n    return a - b\n"  # wrong impl
    tests = [((1, 1), 2)]
    result = evaluator.evaluate(code, "add", tests)
    assert result.outcome == "assertion"
    assert result.ok is False
    assert result.quarantine_path is not None


def test_runtime_exception(evaluator):
    code = "def boom(a, b):\n    return a / 0\n"
    tests = [((1, 1), 0)]
    result = evaluator.evaluate(code, "boom", tests)
    assert result.outcome in {"runtime", "assertion"}  # zero-div before assert


def test_compile_failure(evaluator):
    code = "def broken(:\n    pass"
    result = evaluator.evaluate(code, "broken", [((1,), 1)])
    assert result.outcome == "compile_fail"


def test_ast_violation_short_circuits_before_subprocess(evaluator):
    code = "import os\ndef f(): return os.getcwd()\n"
    result = evaluator.evaluate(code, "f", [((1,), 1)])
    assert result.outcome == "ast_violation"
    # No subprocess started, so no quarantine entry was written for this
    # path; the error message names the disallowed AST node.
    assert "import" in (result.error or "").lower()


def test_invalid_fn_name_is_rejected(evaluator):
    result = evaluator.evaluate("def f(): return 1\n", "not a name", [])
    assert result.outcome == "ast_violation"


def test_timeout_triggers_typed_outcome(tmp_path):
    evaluator = SafeCodeEvaluator(
        timeout_seconds=1.0,
        memory_mb=256,
        quarantine=QuarantineStore(tmp_path / "q"),
    )
    code = (
        "def hang(n):\n"
        "    x = 0\n"
        "    while True:\n"
        "        x += 1\n"
        "    return x\n"
    )
    result = evaluator.evaluate(code, "hang", [((1,), 1)])
    assert result.outcome == "timeout"


def test_evaluation_dict_serialises():
    eval_obj = DiscoveryEvaluation(
        outcome="passed", passed=3, total=3, metadata={"foo": 1}
    )
    payload = eval_obj.to_dict()
    assert payload["outcome"] == "passed"
    assert payload["metadata"]["foo"] == 1


def test_allowed_calls_set_includes_core_helpers():
    """Sanity guard: if someone trims the allowlist by accident, this
    test loudly notices."""
    expected = {"range", "len", "min", "max", "sum", "abs", "sorted"}
    assert expected <= ALLOWED_CALLS


# ---------------------------------------------------------------------------
# stress
# ---------------------------------------------------------------------------
def test_many_tests_run_through_subprocess(evaluator):
    code = "def square(x):\n    return x * x\n"
    tests = [((i,), i * i) for i in range(50)]
    result = evaluator.evaluate(code, "square", tests)
    assert result.outcome == "passed"
    assert result.passed == 50
