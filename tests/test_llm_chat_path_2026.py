################################################################################

"""Phase VI: LLM Backend & Conversation Loop — 2026 Verification Suite

Verifies the critical fixes from Phase VI:
1. LocalLLMAdapter.call URL construction (was NameError)
2. LLMEndpoint is Pydantic BaseModel
3. No print() in state_machine.py
4. AgencyCore resolved from ServiceContainer
5. No bare import requests in llm_router.py
"""
import ast
import os
import sys
import pytest

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SRC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── Fix 1: LLM Router ────────────────────────────────────────

class TestLLMRouterFixes:
    """Verify LLM router critical fixes."""

    def test_llm_endpoint_is_pydantic(self):
        """LLMEndpoint must be a Pydantic BaseModel."""
        from core.brain.llm.llm_router import LLMEndpoint
        from pydantic import BaseModel
        assert issubclass(LLMEndpoint, BaseModel), \
            "LLMEndpoint is not a Pydantic BaseModel"

    def test_llm_endpoint_validation(self):
        """LLMEndpoint should support Pydantic validation and serialization."""
        from core.brain.llm.llm_router import LLMEndpoint, LLMTier
        ep = LLMEndpoint(
            name="test-endpoint",
            tier=LLMTier.PRIMARY,
            endpoint_url="http://localhost:11434",
            model_name="llama3.2"
        )
        assert ep.name == "test-endpoint"
        d = ep.to_dict()
        assert "name" in d
        assert d["tier"] == "primary"
        # model_dump should work
        md = ep.model_dump()
        assert "endpoint_url" in md

    def test_no_dataclass_import_in_router(self):
        """llm_router.py must NOT import dataclass."""
        source = open(os.path.join(SRC_ROOT, "core", "brain", "llm", "llm_router.py")).read()
        assert "from dataclasses import" not in source, \
            "Legacy dataclass import still present in llm_router.py"

    def test_no_sync_requests_import(self):
        """llm_router.py must NOT import sync requests library at top level."""
        source = open(os.path.join(SRC_ROOT, "core", "brain", "llm", "llm_router.py")).read()
        assert "import requests" not in source, \
            "Sync 'import requests' still present in llm_router.py"

    def test_local_adapter_url_defined(self):
        """LocalLLMAdapter.call must construct URL from endpoint_url, not use undefined 'url'."""
        source = open(os.path.join(SRC_ROOT, "core", "brain", "llm", "llm_router.py")).read()
        # The fix adds: url = f"{self.endpoint.endpoint_url}/api/generate"
        assert "self.endpoint.endpoint_url" in source and "/api/generate" in source, \
            "LocalLLMAdapter.call doesn't construct URL from endpoint_url"


# ── Fix 2: StateMachine Cleanup ───────────────────────────────

class TestStateMachineCleanup:
    """Verify StateMachine is production-quality."""

    def test_no_print_statements(self):
        """state_machine.py must NOT contain bare print() calls."""
        source = open(os.path.join(SRC_ROOT, "core", "cognitive", "state_machine.py")).read()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id == "print":
                    pytest.fail(f"Found bare print() in state_machine.py at line {node.lineno}")

    def test_agency_core_from_service_container(self):
        """AgencyCore must be resolved from ServiceContainer, not getattr(orchestrator)."""
        source = open(os.path.join(SRC_ROOT, "core", "cognitive", "state_machine.py")).read()
        assert "ServiceContainer.get(\"agency_core\"" in source, \
            "AgencyCore not resolved from ServiceContainer"
        assert "getattr(self.orchestrator, '_agency_core'" not in source, \
            "Legacy getattr(_agency_core) still present"

    def test_no_none_tool_name_crash(self):
        """_handle_skill must NOT pass None tool_name to _execute_skill_logic."""
        source = open(os.path.join(SRC_ROOT, "core", "cognitive", "state_machine.py")).read()
        assert "_execute_skill_logic(None, {}, user_input)" not in source, \
            "Still passing None tool_name to _execute_skill_logic"

    def test_no_fstring_loggers(self):
        """Logger calls must use %s formatting, not f-strings, for key paths."""
        source = open(os.path.join(SRC_ROOT, "core", "cognitive", "state_machine.py")).read()
        tree = ast.parse(source)
        violations = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                # Check for logger.warning(f"...") and logger.error(f"...")
                if isinstance(func, ast.Attribute) and func.attr in ("warning", "error"):
                    if node.args and isinstance(node.args[0], ast.JoinedStr):
                        violations.append(f"Line {node.lineno}: logger.{func.attr} uses f-string")
        if violations:
            pytest.fail("F-string logger violations:\n" + "\n".join(violations))


##
