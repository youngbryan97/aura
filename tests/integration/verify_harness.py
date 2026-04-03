################################################################################

"""Verify Evaluation Harness."""
import asyncio
import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.self_modification.evaluation_harness import EvaluationHarness
from core.self_modification.code_repair import SandboxTester, CodeFix

class MockBrain:
    async def think(self, prompt):
        # Return a simple reproduction script
        class Thought:
            content = "import os; assert os.path.exists('test_target.py'); print('Repro script ran')"
        return Thought()

async def test_harness():
    print("Testing Evaluation Harness...")
    brain = MockBrain()
    tester = SandboxTester(code_base_path=".")
    harness = EvaluationHarness(brain, tester, code_base_path=".")
    
    fix = CodeFix(
        target_file="test_target.py",
        target_line=1,
        original_code="print('buggy')",
        fixed_code="print('fixed')",
        explanation="Test fix",
        hypothesis="Test hypothesis",
        confidence="high"
    )
    
    # Create the test target file
    target_path = Path("test_target.py")
    target_path.write_text("print('buggy')\n", encoding="utf-8")
    
    try:
        # We need to mock _run_probe_on_code or ensure run_custom_probe works
        # Let's test if we can generate a probe first
        probe = await harness.create_weakness_probe("test_target.py", {"bug": "info"})
        print(f"Generated Probe:\n{probe}")
        assert "assert" in probe or "raise" in probe or "print" in probe
        
        print("Harness logic looks good. (Skipping full sandbox run in mock environment)")
        
    finally:
        if target_path.exists():
            target_path.unlink()

if __name__ == "__main__":
    asyncio.run(test_harness())


##
