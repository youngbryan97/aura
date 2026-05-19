from __future__ import annotations

from pathlib import Path

from tools.audit_degradation import analyze_file


def _write(path: Path, source: str) -> Path:
    path.write_text(source, encoding="utf-8")
    return path


def test_degradation_audit_accepts_multiline_fail_closed_record(tmp_path):
    path = _write(
        tmp_path / "response_generation.py",
        """
from core.runtime.errors import record_degradation

def generate():
    try:
        raise RuntimeError("boom")
    except RuntimeError as exc:
        record_degradation(
            "response_generation",
            exc,
            severity="critical",
        )
        return "safe fallback"
""",
    )

    assert analyze_file(path) == []


def test_degradation_audit_accepts_explicit_recovery_action(tmp_path):
    path = _write(
        tmp_path / "tool_orchestrator.py",
        """
from core.runtime.errors import record_degradation

def execute():
    try:
        raise RuntimeError("boom")
    except RuntimeError as exc:
        record_degradation(
            "tool_orchestrator",
            exc,
            severity="critical",
            action="failed closed and returned explicit tool failure",
        )
        cleanup_side_effect()
""",
    )

    assert analyze_file(path) == []


def test_degradation_audit_flags_bare_limp_on_record(tmp_path):
    path = _write(
        tmp_path / "tool_orchestrator.py",
        """
from core.runtime.errors import record_degradation

def execute():
    try:
        raise RuntimeError("boom")
    except RuntimeError as exc:
        record_degradation("tool_orchestrator", exc)
        cleanup_side_effect()
""",
    )

    issues = analyze_file(path)

    assert len(issues) == 1
    assert issues[0]["severity"] == "CRITICAL"
    assert issues[0]["has_failclose"] is False
    assert issues[0]["has_recovery_action"] is False
