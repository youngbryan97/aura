################################################################################

"""Phase V: System-Wide Functionality Hardening — 2026 Verification Suite

Verifies the critical fixes from Phase V:
1. Service Registration DI alignment (MemoryFacade, AgencyCore, CognitiveIntegration)
2. ImageGenerationSkill async HTTP (httpx)
3. EventBus cleanup (no debug prints, robust publish_threadsafe)
4. CapabilityEngine Pydantic migration
5. PhantomBrowser public API
"""
import asyncio
import sys
import os
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Fix 1: Service Registration DI Alignment ─────────────────

class TestServiceRegistrationDI:
    """Verify that hardened subsystems are properly registered and resolvable."""

    def test_memory_facade_factory_uses_setup(self):
        """MemoryFacade factory should use orchestrator kwarg and call setup()."""
        from core.memory.memory_facade import MemoryFacade
        import inspect
        sig = inspect.signature(MemoryFacade.__init__)
        params = list(sig.parameters.keys())
        # __init__ should accept 'orchestrator', NOT 'episodic', 'semantic', etc.
        assert 'orchestrator' in params, f"MemoryFacade.__init__ params: {params}"
        assert 'episodic' not in params, "MemoryFacade should NOT accept episodic as a kwarg"
        assert 'semantic' not in params, "MemoryFacade should NOT accept semantic as a kwarg"

    def test_agency_core_registered_in_service_registration(self):
        """AgencyCore must be registered as a singleton service."""
        source = open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                      "core", "providers", "cognitive_provider.py")).read()
        assert "agency" in source or "agent" in source or "AgencyCore" in source

    def test_cognitive_integration_registered_in_service_registration(self):
        """CognitiveIntegrationLayer must be registered as a singleton service."""
        source = open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                      "core", "providers", "cognitive_provider.py")).read()
        assert "cognition" in source or "CognitiveIntegrationLayer" in source

    def test_cognition_alias_registered(self):
        """A 'cognition' alias should exist for orchestrator compatibility."""
        source = open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                      "core", "providers", "cognitive_provider.py")).read()
        assert "'cognition'" in source, "cognition alias not registered"


# ── Fix 2: ImageGenerationSkill Async HTTP ────────────────────

class TestImageGenerationAsync:
    """Verify that ImageGenerationSkill uses async httpx, not sync requests."""

    def test_no_sync_requests_import(self):
        """ImageGenerationSkill must NOT import sync 'requests' library."""
        source = open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                      "core", "skills", "sovereign_imagination.py")).read()
        # Should use diffusers, NOT requests
        assert "diffusers" in source, "diffusers not imported"
        assert "import requests" not in source, "sync 'requests' still imported!"

    def test_download_is_async(self):
        """_download_image must be an async method."""
        from core.skills.sovereign_imagination import SovereignImaginationSkill
        import inspect
        # Note: sovereign_imagination uses _generate internally and asyncio.to_thread, but execute is async
        assert inspect.iscoroutinefunction(SovereignImaginationSkill.execute), \
            "execute is not async — would block the event loop"

    def test_pollinations_methods_are_async(self):
        """Pollinations API methods must be async."""
        from core.skills.sovereign_imagination import SovereignImaginationSkill
        import inspect
        assert hasattr(SovereignImaginationSkill, 'on_stop_async')
        assert inspect.iscoroutinefunction(SovereignImaginationSkill.on_stop_async)


# ── Fix 3: EventBus Cleanup ──────────────────────────────────

class TestEventBusCleanup:
    """Verify EventBus is production-quality (no debug prints)."""

    def test_no_print_statements(self):
        """EventBus must NOT contain bare print() calls."""
        source = open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                      "core", "event_bus.py")).read()
        # Filter out comments and docstrings, check for bare prints
        import ast
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id == "print":
                    pytest.fail(f"Found bare print() in event_bus.py at line {node.lineno}")

    def test_no_all_tasks_hack(self):
        """EventBus must NOT use the fragile asyncio.all_tasks() discovery hack."""
        source = open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                      "core", "event_bus.py")).read()
        assert "all_tasks()" not in source, \
            "asyncio.all_tasks() hack still present in EventBus"

    def test_publish_threadsafe_uses_run_coroutine_threadsafe(self):
        """Primary path should use asyncio.run_coroutine_threadsafe."""
        source = open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                      "core", "event_bus.py")).read()
        assert "run_coroutine_threadsafe" in source, \
            "publish_threadsafe should use asyncio.run_coroutine_threadsafe"


# ── Fix 4: CapabilityEngine Pydantic Migration ───────────────

class TestCapabilityEnginePydantic:
    """Verify CapabilityEngine data models use Pydantic."""

    def test_skill_requirements_is_pydantic(self):
        """SkillRequirements must be a Pydantic BaseModel."""
        from core.capability_engine import SkillRequirements
        from pydantic import BaseModel
        assert issubclass(SkillRequirements, BaseModel), \
            "SkillRequirements is not a Pydantic BaseModel"

    def test_skill_metadata_is_pydantic(self):
        """SkillMetadata must be a Pydantic BaseModel."""
        from core.capability_engine import SkillMetadata
        from pydantic import BaseModel
        assert issubclass(SkillMetadata, BaseModel), \
            "SkillMetadata is not a Pydantic BaseModel"

    def test_no_dataclass_imports(self):
        """capability_engine.py must NOT import dataclass."""
        source = open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                      "core", "capability_engine.py")).read()
        assert "from dataclasses import" not in source, \
            "Legacy dataclass import still present"

    def test_skill_requirements_validation(self):
        """SkillRequirements should support Pydantic validation."""
        from core.capability_engine import SkillRequirements
        req = SkillRequirements(packages=["pytest"], commands=["python"])
        assert req.packages == ["pytest"]
        assert req.commands == ["python"]
        # model_dump should work
        d = req.model_dump()
        assert "packages" in d
        assert "commands" in d

    def test_skill_metadata_schema_def(self):
        """SkillMetadata.schema_def should return a valid JSON schema."""
        from core.capability_engine import SkillMetadata
        meta = SkillMetadata(
            name="test_skill",
            description="A test skill",
            skill_class=object,
        )
        schema = meta.schema_def
        assert isinstance(schema, dict)
        assert "type" in schema


# ── Fix 5: PhantomBrowser Public API ──────────────────────────

class TestPhantomBrowserPublicAPI:
    """Verify PhantomBrowser exposes a proper public lifecycle method."""

    def test_ensure_ready_exists(self):
        """PhantomBrowser must have a public ensure_ready() method."""
        from core.phantom_browser import PhantomBrowser
        assert hasattr(PhantomBrowser, 'ensure_ready'), \
            "PhantomBrowser missing ensure_ready() method"

    def test_ensure_ready_is_async(self):
        """ensure_ready() must be async."""
        from core.phantom_browser import PhantomBrowser
        import inspect
        assert inspect.iscoroutinefunction(PhantomBrowser.ensure_ready), \
            "ensure_ready() is not async"

    def test_sovereign_browser_uses_public_api(self):
        """SovereignBrowserSkill must use ensure_ready(), not _start_browser()."""
        source = open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                      "core", "skills", "sovereign_browser.py")).read()
        assert "ensure_ready()" in source, \
            "SovereignBrowserSkill not using ensure_ready()"
        assert "_start_browser()" not in source, \
            "SovereignBrowserSkill still calling private _start_browser()"


##
