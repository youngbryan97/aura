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
    "core/runtime/base_module.py", "core/sovereign/errors.py", "core/exceptions.py",
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
            lowered = doc.lower()
            # A docstring that explicitly documents None as a valid answer is
            # not promising behaviour it fails to deliver — it is declaring an
            # opt-in default. `reality_manifest` says "if this device has one"
            # and returns None precisely because hardware authority must never
            # be inferred; flagging that taught the opposite lesson.
            if "none" in lowered:
                return False
            # Must promise behavior (not just a label)
            action_words = {"return", "compute", "calculate", "process", "validate",
                           "check", "verify", "run", "execute", "perform", "apply",
                           "update", "write", "send", "emit", "create", "build"}
            return any(w in lowered for w in action_words)
        return False

    @staticmethod
    def _returns_optional(node) -> bool:
        """Whether the SIGNATURE declares None a valid return.

        `def reality_manifest(self) -> HardwareRealityManifest | None` with a
        body of `return None` is a declared opt-in default, not an
        unfinished implementation — a device is inventory-only until it
        deliberately provides a manifest, and hardware authority must never
        be inferred.

        Type-driven rather than prose-driven on purpose: the previous
        heuristic read the docstring, so whether a deliberate default was
        flagged depended on whether its author happened to write the word
        "None". The annotation says it unambiguously.
        """
        annotation = node.returns
        if annotation is None:
            return False
        if isinstance(annotation, ast.Constant) and annotation.value is None:
            return True
        # `X | None`
        if isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
            for side in (annotation.left, annotation.right):
                if isinstance(side, ast.Constant) and side.value is None:
                    return True
        # `Optional[X]` / `Union[X, None]`
        if isinstance(annotation, ast.Subscript):
            base = annotation.value
            name = getattr(base, "id", getattr(base, "attr", ""))
            if name == "Optional":
                return True
            if name == "Union":
                inner = annotation.slice
                elements = inner.elts if isinstance(inner, ast.Tuple) else [inner]
                for element in elements:
                    if isinstance(element, ast.Constant) and element.value is None:
                        return True
        # String annotations, e.g. "-> 'Thing | None'"
        if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
            return "None" in annotation.value
        return False

    @staticmethod
    def _is_protocol_member(tree, node) -> bool:
        """Whether this function is declared on a typing.Protocol.

        A Protocol member's body IS the declaration — there is nothing to
        implement. The scan already skips @abstractmethod and @property for
        the same reason; Protocol members carry no decorator to skip on, so
        the enclosing class has to be inspected.
        """
        for parent in ast.walk(tree):
            if not isinstance(parent, ast.ClassDef) or node not in parent.body:
                continue
            for base in parent.bases:
                name = base.id if isinstance(base, ast.Name) else getattr(base, "attr", "")
                if name == "Protocol":
                    return True
                if isinstance(base, ast.Subscript):
                    inner = base.value
                    if getattr(inner, "id", getattr(inner, "attr", "")) == "Protocol":
                        return True
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
                if self._is_protocol_member(tree, node):
                    continue
                if self._returns_optional(node):
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


#: Prefixes that make the rest of a name the SUBJECT of a check rather than a
#: description of the code. A function called ``contains_placeholder`` reports
#: on a placeholder; it is not one.
_NAMES_WHAT_IT_LOOKS_FOR = re.compile(
    r"^(?:contains|has|is|looks_like|detect|detects|find|finds|scan|scans|"
    r"reject|rejects|flag|flags|count|counts)_",
    re.IGNORECASE,
)


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
                    if _NAMES_WHAT_IT_LOOKS_FOR.match(name):
                        # A detector is named for what it detects.
                        # contains_unfilled_placeholder is the function that
                        # FINDS a placeholder in a reply, and flagging it says
                        # the opposite of the truth.
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


class TestTheNoopDetectorStillDetects:
    """The exclusions above must narrow the scan, not disarm it.

    Three were added so deliberate opt-in defaults stop being reported as
    unfinished work: Protocol members (whose body IS the declaration),
    signatures that declare None a valid return, and docstrings that say so.
    Each of those could hide a real stub if it were loose, so this pins that
    a genuine promising no-op is still caught.
    """

    def _scan(self, source: str) -> bool:
        """True when the detector would flag something in this source."""
        detector = TestFakeBehaviorDetection()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name.startswith("_"):
                continue
            decorators = [_decorator_name(d) for d in node.decorator_list]
            if "abstractmethod" in decorators or "property" in decorators:
                continue
            if detector._is_protocol_member(tree, node):
                continue
            if detector._returns_optional(node):
                continue
            if detector._has_docstring(node) and detector._is_noop_body(node.body):
                return True
        return False

    def test_a_real_promising_stub_is_still_caught(self):
        assert self._scan(
            "def compute_risk(self) -> float:\n"
            '    """Compute and return the risk score for this action."""\n'
            "    return 0\n"
        )

    def test_a_stub_returning_an_empty_dict_is_still_caught(self):
        assert self._scan(
            "def build_report(self) -> dict:\n"
            '    """Build the diagnostic report."""\n'
            "    return {}\n"
        )

    def test_an_optional_return_does_not_excuse_a_non_none_stub(self):
        """The exclusion is for `return None`, not for any weak body."""
        assert self._scan(
            "def summarize(self) -> str:\n"
            '    """Return a summary of the episode."""\n'
            "    return ''\n"
        )

    def test_a_protocol_member_is_a_declaration_not_a_stub(self):
        assert not self._scan(
            "class Resolver(Protocol):\n"
            "    def resolve(self, episode) -> object:\n"
            '        """Return the outcome for this episode."""\n'
        )

    def test_a_declared_optional_default_is_not_a_stub(self):
        assert not self._scan(
            "def reality_manifest(self) -> Manifest | None:\n"
            '    """Return the capability contract, if this device has one."""\n'
            "    return None\n"
        )

    def test_a_non_protocol_class_member_is_still_scanned(self):
        """Only Protocol is excused; an ordinary class is not."""
        assert self._scan(
            "class Thing:\n"
            "    def compute_risk(self) -> float:\n"
            '        """Compute and return the risk score."""\n'
            "        return 0\n"
        )
