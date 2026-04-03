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
            self.test_file.unlink()

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

if __name__ == "__main__":
    # Run test
    async def run():
        suite = unittest.TestLoader().loadTestsFromTestCase(TestFixPersistence)
        runner = unittest.TextTestRunner()
        runner.run(suite)
    # asyncio.run(run())
