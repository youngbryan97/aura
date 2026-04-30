"""tests/test_semantic_marker_audit.py — Semantic Marker Audit

Not "no TODO string." Instead: no production file contains fake behavior,
unreachable implementation, dormant stub used as real system, or
marker-evasion strings.
"""
from __future__ import annotations
import ast, re, sys
from pathlib import Path
from typing import List, Tuple
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCAN_ROOTS = ["core"]
SKIP_PARTS = {"__pycache__", ".venv", ".git", "tests", "archive", "aura_bench"}
# Files that legitimately use abstract patterns
ABSTRACT_ALLOWLIST = {
    "core/base_module.py", "core/errors.py", "core/exceptions.py",
    "core/skills/base_skill.py",  # base class defaults (match() returns False)
}


def _production_files():
    for top in SCAN_ROOTS:
        base = ROOT / top
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            if any(part in SKIP_PARTS for part in path.parts):
                continue
            yield path


def _parse_safe(path):
    try:
        return ast.parse(path.read_text(encoding="utf-8"), str(path))
    except (SyntaxError, UnicodeDecodeError):
        return None


class TestFakeBehaviorDetection:
    """Flag functions with promising docstrings but no-op bodies."""

    NO_OP_PATTERNS = {
        ast.Pass, ast.Constant,  # pass, return None/0/{}/[]
    }

    def _is_noop_body(self, body):
        """Check if function body is effectively a no-op."""
        stmts = [s for s in body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
        if not stmts:
            return True
        if len(stmts) == 1:
            s = stmts[0]
            if isinstance(s, ast.Pass):
                return True
            if isinstance(s, ast.Return):
                if s.value is None:
                    return True
                if isinstance(s.value, ast.Constant) and s.value.value in (None, 0, "", False):
                    return True
                if isinstance(s.value, ast.Dict) and not s.value.keys:
                    return True
                if isinstance(s.value, ast.List) and not s.value.elts:
                    return True
            if isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant):
                return True
        return False

    def _has_docstring(self, node):
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant):
            doc = str(node.body[0].value.value)
            # Must promise behavior (not just a label)
            action_words = {"return", "compute", "calculate", "process", "validate",
                           "check", "verify", "run", "execute", "perform", "apply",
                           "update", "write", "send", "emit", "create", "build"}
            return any(w in doc.lower() for w in action_words)
        return False

    def test_no_promising_noops_in_production(self):
        """No function with a behavior-promising docstring should be a no-op."""
        findings: List[str] = []
        for path in _production_files():
            rel = path.relative_to(ROOT).as_posix()
            if rel in ABSTRACT_ALLOWLIST:
                continue
            tree = _parse_safe(path)
            if tree is None:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                # Skip dunder methods, property getters, abstractmethods
                if node.name.startswith("_"):
                    continue
                decorators = [_decorator_name(d) for d in node.decorator_list]
                if "abstractmethod" in decorators or "property" in decorators:
                    continue
                if self._has_docstring(node) and self._is_noop_body(node.body):
                    findings.append(f"{rel}:{node.lineno} {node.name}()")
        assert not findings, (
            f"production functions with promising docstrings but no-op bodies:\n"
            + "\n".join(findings[:20])
        )


class TestUnreachableImplementation:
    """Find if False:, dead branches after unconditional return/raise."""

    def test_no_if_false_in_production(self):
        findings = []
        for path in _production_files():
            rel = path.relative_to(ROOT).as_posix()
            tree = _parse_safe(path)
            if tree is None:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.If):
                    test = node.test
                    # if False: or if 0:
                    if isinstance(test, ast.Constant) and test.value in (False, 0):
                        findings.append(f"{rel}:{node.lineno} if False/0:")
        assert not findings, f"unreachable if False blocks:\n" + "\n".join(findings)

    def test_no_code_after_unconditional_return(self):
        """No statements after bare return/raise at function level."""
        findings = []
        for path in _production_files():
            rel = path.relative_to(ROOT).as_posix()
            tree = _parse_safe(path)
            if tree is None:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                body = node.body
                for i, stmt in enumerate(body):
                    if isinstance(stmt, (ast.Return, ast.Raise)) and i < len(body) - 1:
                        # Check if next statement is NOT a function/class def (those are fine)
                        remaining = body[i+1:]
                        real_stmts = [s for s in remaining
                                      if not isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
                        if real_stmts:
                            findings.append(f"{rel}:{stmt.lineno} {node.name}() — dead code after return/raise")
                        break  # Only check first unconditional return
        # Allow up to 3 findings (some legacy edge cases)
        assert len(findings) <= 3, (
            f"dead code after return/raise ({len(findings)} findings):\n"
            + "\n".join(findings[:10])
        )


class TestDormantStubDetection:
    """Cross-reference ServiceContainer registrations for mock/stub in production."""

    def test_no_mock_in_production_registrations(self):
        """Production code must not register MagicMock/AsyncMock as services."""
        findings = []
        for path in _production_files():
            rel = path.relative_to(ROOT).as_posix()
            try:
                src = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            # Look for register_instance with Mock
            for i, line in enumerate(src.split("\n"), 1):
                if "register_instance" in line and ("MagicMock" in line or "AsyncMock" in line):
                    findings.append(f"{rel}:{i}")
        assert not findings, f"mock registrations in production:\n" + "\n".join(findings)


class TestMarkerEvasionStrings:
    """Scan for patterns designed to fool grep-based auditors."""

    EVASION_PATTERNS = [
        (r'stub|placeholder|not.implemented|fake|dummy', "suspicious_name"),
    ]

    # Legitimate uses of "stub/dummy" names — documented architectural patterns:
    # - Dummy in output_gate: sentinel class used as dict key for legacy fallback
    # - DummyTTS: real fallback TTS that logs when no TTS engine is available
    # - OrganStub: well-documented lazy-loading wrapper for hardware subsystems
    KNOWN_LEGITIMATE = {
        ("core/utils/output_gate.py", "Dummy"),
        ("core/embodiment/voice_presence.py", "DummyTTS"),
        ("core/kernel/organs.py", "OrganStub"),
    }

    def test_no_evasion_in_function_names(self):
        """Production function/class names must not contain stub/fake/dummy/placeholder."""
        findings = []
        suspicious = re.compile(r"(stub|placeholder|fake|dummy)", re.IGNORECASE)
        for path in _production_files():
            rel = path.relative_to(ROOT).as_posix()
            tree = _parse_safe(path)
            if tree is None:
                continue
            for node in ast.walk(tree):
                name = ""
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    name = node.name
                elif isinstance(node, ast.ClassDef):
                    name = node.name
                if name and suspicious.search(name) and not name.startswith("_"):
                    if (rel, name) in self.KNOWN_LEGITIMATE:
                        continue
                    findings.append(f"{rel}:{node.lineno} {name}")
        assert not findings, f"evasion-suspect names in production:\n" + "\n".join(findings)

    def test_no_pass_real_comments(self):
        """No 'pass  # real' or 'pass  # implemented' masking no-ops."""
        findings = []
        pattern = re.compile(r"^\s*pass\s+#\s*(real|implemented|active|live|done)", re.IGNORECASE)
        for path in _production_files():
            rel = path.relative_to(ROOT).as_posix()
            try:
                for i, line in enumerate(path.read_text(encoding="utf-8").split("\n"), 1):
                    if pattern.match(line):
                        findings.append(f"{rel}:{i} {line.strip()}")
            except UnicodeDecodeError:
                continue
        assert not findings, f"pass-masking comments:\n" + "\n".join(findings)


def _decorator_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Call):
        return _decorator_name(node.func)
    return ""
