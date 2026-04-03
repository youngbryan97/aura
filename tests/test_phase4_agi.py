import pytest
import sys
import os
from core.capability_engine import CapabilityEngine, Sandbox2
from core.resilience.hotfix_engine import HotfixEngine



@pytest.mark.asyncio
async def test_hotfix_engine_reload():
    """Verify that the HotfixEngine can reload a module."""
    # Ensure module is loaded
    import core.utils.sanitizer
    
    engine = HotfixEngine()
    module_name = "core.utils.sanitizer"
    
    result = await engine.reload_module(module_name)
    assert result["ok"] is True, f"Hotfix failed: {result.get('error')}"
    assert result["reloaded"] == module_name
    assert engine.reloads_total == 1
