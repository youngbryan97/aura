from core.runtime.atomic_writer import atomic_write_text
import asyncio
import unittest
from pathlib import Path
from core.self_modification.code_repair import AutonomousCodeRepair
from core.self_modification.error_intelligence import CodeFix

class TestTypeSafeRepair(unittest.TestCase):
    def setUp(self):
        self.repair = AutonomousCodeRepair()

    async def test_pyright_guard_rejection(self):
        # Create a mock fix that has a type error
        fix = CodeFix(
            target_file="core/test_type_error.py",
            original_code="def foo(x: int) -> int:\n    return x",
            fixed_code="def foo(x: int) -> int:\n    return 'string'  # Type error!",
            explanation="Introducing a type error for testing"
        )
        
        # We need a temporary sandbox
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            sandbox = Path(tmpdir)
            test_file = sandbox / fix.target_file
            test_file.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(test_file, fix.fixed_code)
            
            # Run tests in sandbox
            results = await self.repair._run_tests_in_sandbox(sandbox, fix)
            
            # Should fail due to Pyright mismatch
            self.assertFalse(results["success"])
            self.assertTrue(any("Pyright" in e or "type mismatch" in e.lower() for e in results["errors"]))

if __name__ == "__main__":
    # Use nest_asyncio or just run with asyncio
    import asyncio
    async def run_test():
        suite = unittest.TestLoader().loadTestsFromTestCase(TestTypeSafeRepair)
        runner = unittest.TextTestRunner()
        runner.run(suite)
    # asyncio.run(run_test()) # This might hang in some envs, so we'll just use simple assert in a script
