"""Tests for the typed mutation evaluator + quarantine."""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from core.self_modification.mutation_safety import (
    MutationOutcome,
    QuarantineStore,
    SafeMutationEvaluator,
)


@pytest.fixture
def evaluator(tmp_path: Path) -> SafeMutationEvaluator:
    quarantine = QuarantineStore(tmp_path / "quarantine")
    return SafeMutationEvaluator(
        timeout_seconds=10.0,
        memory_mb=256,
        quarantine=quarantine,
    )


# ---------------------------------------------------------------------------
# typed outcomes
# ---------------------------------------------------------------------------
def test_passed_for_clean_module(evaluator):
    diag = evaluator.evaluate("def add(a, b):\n    return a + b\n")
    assert diag.outcome is MutationOutcome.PASSED
    assert diag.exit_code == 0
    assert diag.quarantine_path is None


def test_compile_fail_on_syntax_error(evaluator):
    diag = evaluator.evaluate("def broken(:\n    pass\n")
    assert diag.outcome is MutationOutcome.COMPILE_FAIL
    assert diag.quarantine_path is not None
    assert "SyntaxError" in diag.traceback_text


def test_import_fail(evaluator):
    diag = evaluator.evaluate("import definitely_not_a_real_pkg_xyz_42\n")
    assert diag.outcome is MutationOutcome.IMPORT_FAIL
    assert diag.quarantine_path is not None


def test_runtime_exception(evaluator):
    diag = evaluator.evaluate(
        textwrap.dedent(
            """
            def boom():
                return 1 / 0
            boom()
            """
        )
    )
    assert diag.outcome is MutationOutcome.RUNTIME_EXCEPTION
    assert "ZeroDivisionError" in diag.traceback_text


def test_assertion_fail_in_module_body(evaluator):
    diag = evaluator.evaluate("assert 1 == 2, 'nope'\n")
    assert diag.outcome is MutationOutcome.ASSERTION_FAIL
    assert "AssertionError" in diag.traceback_text


def test_assertion_fail_in_test(evaluator):
    diag = evaluator.evaluate(
        "def add(a, b): return a + b\n",
        test_source="assert add(1, 1) == 3\n",
    )
    assert diag.outcome is MutationOutcome.ASSERTION_FAIL
    assert diag.quarantine_path is not None


def test_test_passes(evaluator):
    diag = evaluator.evaluate(
        "def add(a, b): return a + b\n",
        test_source="assert add(2, 2) == 4\n",
    )
    assert diag.outcome is MutationOutcome.PASSED


def test_timeout(tmp_path):
    evaluator = SafeMutationEvaluator(
        timeout_seconds=1.0,
        memory_mb=256,
        quarantine=QuarantineStore(tmp_path / "q"),
    )
    diag = evaluator.evaluate(
        textwrap.dedent(
            """
            import time
            time.sleep(60)
            """
        )
    )
    assert diag.outcome is MutationOutcome.TIMEOUT
    assert diag.quarantine_path is not None
    assert diag.runtime_seconds < 5.0  # sanity: parent killed the child quickly


def test_parent_does_not_crash_on_malformed_mutation(evaluator):
    """The whole point of the typed evaluator: a malformed mutation
    must degrade to a diagnostic, never crash the parent process."""
    catastrophic_sources = [
        "def \xff broken_token():\n    pass\n",  # not valid utf-8 token
        "raise SystemExit('bye')\n",
        "import sys; sys.exit(99)\n",
        "1/0\n",
        "assert False\n",
        "this is not python at all !!!\n",
        "while True: pass\n",  # would hang without a small timeout
    ]
    fast_evaluator = SafeMutationEvaluator(
        timeout_seconds=2.0,
        memory_mb=256,
        quarantine=evaluator.quarantine,
    )
    outcomes = []
    for src in catastrophic_sources:
        diag = fast_evaluator.evaluate(src)
        outcomes.append(diag.outcome)
        # Either way, the parent kept running — that is the assertion.
    # And every catastrophic input ended with a typed (non-PASSED) outcome.
    assert MutationOutcome.PASSED not in outcomes


# ---------------------------------------------------------------------------
# quarantine
# ---------------------------------------------------------------------------
def test_quarantine_layout(evaluator, tmp_path):
    diag = evaluator.evaluate("def f():\n    return 1/0\nf()\n")
    assert diag.quarantine_path is not None
    entry = Path(diag.quarantine_path)
    assert entry.exists()
    assert (entry / "source.py").exists()
    assert (entry / "result.json").exists()
    assert (entry / "stdout.log").exists()
    assert (entry / "stderr.log").exists()
    payload = json.loads((entry / "result.json").read_text(encoding="utf-8"))
    assert payload["outcome"] == "runtime_exception"


def test_passed_does_not_quarantine(evaluator):
    diag = evaluator.evaluate("x = 1\n")
    assert diag.outcome is MutationOutcome.PASSED
    assert diag.quarantine_path is None
    # Quarantine root may have been created but contains nothing.
    entries = evaluator.quarantine.list_entries()
    assert entries == []


def test_quarantine_is_per_invocation(evaluator):
    # Two different broken mutations get two different quarantine dirs.
    a = evaluator.evaluate("syntax error here:::")
    b = evaluator.evaluate("import not_a_real_module_zzz")
    assert a.quarantine_path != b.quarantine_path
    entries = evaluator.quarantine.list_entries()
    assert len(entries) == 2


# ---------------------------------------------------------------------------
# diagnostics shape
# ---------------------------------------------------------------------------
def test_diagnostics_has_required_fields(evaluator):
    diag = evaluator.evaluate("syntax_error_here !!!:")
    d = diag.to_dict()
    assert d["outcome"] == "compile_fail"
    assert "runtime_seconds" in d
    assert "exit_code" in d
    assert "stdout" in d
    assert "stderr" in d
    assert d["quarantine_path"] is not None


def test_outcome_enum_completeness():
    expected = {
        "passed",
        "compile_fail",
        "import_fail",
        "runtime_exception",
        "assertion_fail",
        "timeout",
        "oom",
    }
    assert {o.value for o in MutationOutcome} == expected
