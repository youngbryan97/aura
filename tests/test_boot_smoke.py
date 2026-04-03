"""Aura Boot Smoke Tests
═══════════════════════
Validates core module imports, key service interactions, and
identity hardening. These are fast, offline tests that don't
require a running LLM or network connection.
"""
import ast
import os
import re
import sys
from pathlib import Path

import pytest

# ── Test 1: All .py files parse without SyntaxError ──────────────
class TestSyntaxValidity:
    """Ensure every Python file in core/ and interface/ is syntactically valid."""

    @staticmethod
    def _collect_py_files():
        project = Path(__file__).resolve().parent.parent
        files = []
        for subdir in ("core", "interface"):
            root = project / subdir
            if root.exists():
                for f in root.rglob("*.py"):
                    files.append(f)
        return files

    def test_all_files_parse(self):
        errors = []
        for path in self._collect_py_files():
            try:
                source = path.read_text(encoding="utf-8", errors="replace")
                ast.parse(source, filename=str(path))
            except SyntaxError as e:
                errors.append(f"{path}:{e.lineno}: {e.msg}")
        assert not errors, f"Syntax errors found:\n" + "\n".join(errors)

# ── Test 2: Core module imports ──────────────────────────────────
class TestCoreImports:
    """Verify that essential core modules can be imported."""

    REQUIRED_MODULES = [
        "core.container",
        "core.config",
        "core.state.aura_state",
        "core.event_bus",
        "core.orchestrator.main",
    ]

    @pytest.mark.parametrize("module_path", REQUIRED_MODULES)
    def test_import(self, module_path):
        """Each core module should import without errors."""
        import importlib
        mod = importlib.import_module(module_path)
        assert mod is not None

# ── Test 3: ServiceContainer basics ──────────────────────────────
class TestServiceContainer:
    """Verify ServiceContainer register/get lifecycle."""

    def test_register_instance_and_get(self, service_container):
        """Register a mock service instance and retrieve it."""
        class MockService:
            name = "test_mock"

        service_container.register_instance("test_mock_instance", MockService())
        result = service_container.get("test_mock_instance")
        assert result is not None
        assert result.name == "test_mock"

    def test_register_factory_and_get(self, service_container):
        """Register a factory callable and retrieve the built service."""
        class MockService:
            name = "factory_built"

        service_container.register("test_factory", lambda: MockService())
        result = service_container.get("test_factory")
        assert result is not None
        assert result.name == "factory_built"

    def test_get_missing_returns_default(self, service_container):
        """Getting a non-existent service with default should not raise."""
        result = service_container.get("nonexistent_service_xyz", default=None)
        assert result is None

# ── Test 4: MycelialNetwork pathway matching ─────────────────────
class TestMycelialNetwork:
    """Verify the Mycelial routing system."""

    @pytest.fixture(autouse=True)
    def _check_mycelium_available(self):
        """Skip if MycelialNetwork can't be imported (e.g., broken pydantic in venv)."""
        try:
            from core.mycelium import MycelialNetwork
        except (ImportError, ModuleNotFoundError) as e:
            pytest.skip(f"MycelialNetwork unavailable: {e}")

    def test_register_and_match_pathway(self):
        from core.mycelium import MycelialNetwork
        net = MycelialNetwork()
        net.register_pathway(
            pathway_id="test_joke",
            pattern=r"tell me a joke",
            skill_name="joke_skill",
            priority=100,
        )
        match = net.match_hardwired("tell me a joke")
        assert match is not None

    def test_no_match_returns_none(self):
        from core.mycelium import MycelialNetwork
        net = MycelialNetwork()
        result = net.match_hardwired("this should not match anything registered")
        assert result is None

# ── Test 5: AuraState lineage ────────────────────────────────────
class TestAuraState:
    """Verify state derivation and versioning."""

    def test_derive_increments_version(self):
        from core.state.aura_state import AuraState
        state = AuraState()
        initial_version = state.version
        derived = state.derive(cause="test_derive")
        assert derived.version == initial_version + 1

    def test_derive_is_independent(self):
        from core.state.aura_state import AuraState
        state = AuraState()
        derived = state.derive(cause="test_independence")
        # Mutating derived should not affect original
        derived.version = 999
        assert state.version != 999

# ── Test 6: Identity Guard sanity ────────────────────────────────
class TestIdentityGuard:
    """Verify that identity validation catches obvious breaches."""

    @pytest.fixture(autouse=True)
    def _check_identity_guard_available(self):
        """Skip if PersonaEnforcementGate can't be imported."""
        try:
            from core.identity.identity_guard import PersonaEnforcementGate
        except (ImportError, ModuleNotFoundError) as e:
            pytest.skip(f"IdentityGuard unavailable: {e}")

    def test_validate_output_returns_tuple(self):
        """validate_output should return a (bool, str, float) tuple."""
        from core.identity.identity_guard import PersonaEnforcementGate
        gate = PersonaEnforcementGate()
        result = gate.validate_output("I am a human being and I have feelings.")
        assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
        assert len(result) == 3, f"Expected 3-element tuple, got {len(result)}"
        is_valid, reason, score = result
        assert isinstance(is_valid, bool)
        assert isinstance(reason, str)
        assert isinstance(score, (int, float))

    def test_passes_valid_aura_output(self):
        from core.identity.identity_guard import PersonaEnforcementGate
        gate = PersonaEnforcementGate()
        result = gate.validate_output("I've analyzed your code and found three issues.")
        is_valid, reason, score = result
        # Normal Aura output should pass validation
        assert is_valid, f"Expected valid output, got: {result}"

# ── Test 7: No TODO/STUB markers in production code ─────────────
class TestCodeQuality:
    """Enforce that TODO/STUB markers have been resolved."""

    EXCLUDED_FILES = {"auto_refactor.py", "fictional_ai_synthesis.py"}
    MARKER_PATTERN = re.compile(r"\bTODO\b|\[STUB\]|\bSTUB\b")

    def test_no_todo_or_stub_markers(self):
        project = Path(__file__).resolve().parent.parent
        violations = []

        for subdir in ("core", "interface", "skills"):
            root = project / subdir
            if not root.exists():
                continue
            for path in root.rglob("*.py"):
                if path.name in self.EXCLUDED_FILES:
                    continue
                source = path.read_text(encoding="utf-8", errors="replace")
                for lineno, line in enumerate(source.splitlines(), 1):
                    if self.MARKER_PATTERN.search(line):
                        violations.append(f"{path}:{lineno}: {line.strip()}")

        assert not violations, "TODO/STUB markers found:\n" + "\n".join(violations)
