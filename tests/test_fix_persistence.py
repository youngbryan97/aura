import asyncio
import unittest
from unittest.mock import AsyncMock
from pathlib import Path
import json
from core.self_modification.self_modification_engine import AutonomousSelfModificationEngine
from core.self_modification.code_repair import CodeFix
from core.config import config

class TestFixPersistence(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # We need a mock cognitive engine
        class MockBrain:
            async def think(self, prompt, priority=0.0):
                return type('Thought', (), {'content': 'FIXED'})()
        
        self.engine = AutonomousSelfModificationEngine(MockBrain())
        self.test_file = Path("core/temp_fix_test.py")
        self.test_file.write_text("def old_function():\n    return 'old'")
        self.sepsis_registry = config.paths.data_dir / "sepsis_registry.json"
        if self.sepsis_registry.exists():
            try:
                data = json.loads(self.sepsis_registry.read_text())
                banned = [p for p in data.get("banned_files", []) if p != str(self.test_file)]
                data["banned_files"] = banned
                self.sepsis_registry.write_text(json.dumps(data))
            except Exception:
                pass

    def tearDown(self):
        if self.test_file.exists():
            get_task_tracker().create_task(get_storage_gateway().delete(self.test_file, cause='TestFixPersistence.tearDown'))

    async def test_permanent_fix_application(self):
        # Create a mock fix proposal
        fix = CodeFix(
            target_file=str(self.test_file),
            target_line=1,
            original_code="def old_function():\n    return 'old'",
            fixed_code="def new_function():\n    return 'new'",
            explanation="Test permanent fix",
            hypothesis="Testing persistence",
            confidence="high"
        )
        
        proposal = {
            "bug": {"pattern": {"events": [{"error_type": "test_persistence"}]}},
            "fix": fix,
            "test_results": {"success": True}
        }
        
        # Bypass swarm for testing
        self.engine._swarm_review = AsyncMock(return_value=True)
        
        # Apply fix
        success = await self.engine.apply_fix(proposal, force=True)
        
        self.assertTrue(success)
        
        # Verify file was actually modified
        content = self.test_file.read_text()
        self.assertIn("def new_function()", content)
        self.assertNotIn("def old_function()", content)

    async def test_apply_fix_resolves_subsystem_and_incident(self):
        from core.runtime.errors import get_subsystem_registry
        from core.resilience.incident_manager import get_incident_manager, IncidentSeverity, IncidentStatus
        from core.self_modification.error_intelligence import ErrorPattern, ErrorEvent
        
        # Setup subsystem status to degraded
        subsystem_reg = get_subsystem_registry()
        health = subsystem_reg.register("test_sme_subsystem")
        health.mark_degraded("simulated failure")
        
        # Setup active incident
        incident_mgr = get_incident_manager()
        category = "degradation:test_sme_subsystem"
        incident = incident_mgr.report(
            category=category,
            description="simulated failure",
            severity=IncidentSeverity.DEGRADED
        )
        
        # Mock fix proposal with full ErrorPattern / ErrorEvent
        event = ErrorEvent(
            timestamp=123.45,
            error_type="ValueError",
            error_message="Oops",
            stack_trace="Traceback",
            context={"subsystem": "test_sme_subsystem"}
        )
        pattern = ErrorPattern(
            fingerprint="test_fingerprint",
            occurrences=1,
            first_seen=123.45,
            last_seen=123.45,
            events=[event],
            severity="high"
        )
        
        fix = CodeFix(
            target_file=str(self.test_file),
            target_line=1,
            original_code="def old_function():\n    return 'old'",
            fixed_code="def new_function():\n    return 'new'",
            explanation="Test repair",
            hypothesis="Testing resolution",
            confidence="high"
        )
        
        proposal = {
            "bug": {"pattern": pattern},
            "fix": fix,
            "test_results": {"success": True}
        }
        
        self.engine._swarm_review = AsyncMock(return_value=True)
        success = await self.engine.apply_fix(proposal, force=True)
        self.assertTrue(success)
        
        # Subsystem should now be healthy and incident resolved!
        self.assertEqual(health.status, "healthy")
        self.assertEqual(incident.status, IncidentStatus.RECOVERED)


if __name__ == "__main__":
    # Run test
    async def run():
        suite = unittest.TestLoader().loadTestsFromTestCase(TestFixPersistence)
        runner = unittest.TextTestRunner()
        runner.run(suite)
    # asyncio.run(run())
