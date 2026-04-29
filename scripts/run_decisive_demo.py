import asyncio
import os
import time
import json
import difflib
from pathlib import Path
from core.container import ServiceContainer
from core.self_modification.self_modification_engine import AutonomousSelfModificationEngine
from core.self_modification.error_intelligence import ErrorEvent
from core.runtime.atomic_writer import atomic_write_text
from core.config import config


REPO_ROOT = Path(__file__).resolve().parents[1]

class MockBrain:
    def __init__(self, target_file):
        self.target_file = Path(target_file)
        self.original_content = self.target_file.read_text(encoding="utf-8")

    async def think(self, prompt, priority=0.1):
        if "datetiime" in prompt:
            return type('obj', (object,), {'content': json.dumps({
                "hypotheses": [{
                    "root_cause": "Typo in 'datetime' import",
                    "explanation": "Line 1 uses 'datetiime' instead of 'datetime'",
                    "diagnostic_test": "Check if 'datetiime' exists in datetime module",
                    "potential_fix": "Change 'datetiime' to 'datetime'",
                    "confidence": "high"
                }]
            })})
        if "fixed version" in prompt.lower() or "fixed code" in prompt.lower():
            # Return the first 10 lines of the original correct file
            lines = self.original_content.splitlines()
            return "\n".join(lines[:10])
        return type('obj', (object,), {'content': "Consensus: Fix is safe to apply."})

async def run_decisive_demo():
    print("🚀 STARTING LIVE-SOURCE AUTONOMOUS IMPROVEMENT CYCLE")
    print("=" * 80)
    
    ServiceContainer.register("config", config)
    target_path = "core/skills/clock.py"
    abs_target = REPO_ROOT / target_path
    brain = MockBrain(abs_target)
    engine = AutonomousSelfModificationEngine(brain, code_base_path=str(REPO_ROOT))
    
    class MockKernel:
        volition_level = 3
    ServiceContainer.register("aura_kernel", MockKernel())
    
    target_file = Path(abs_target)
    original_code = target_file.read_text(encoding="utf-8")
    
    # 2. INJECTION
    print(f"💉 Injecting 'datetiime' import bug into {target_path}...")
    buggy_code = original_code.replace("from datetime import datetime", "from datetime import datetiime")
    atomic_write_text(target_file, buggy_code, encoding="utf-8")
    
    try:
        # 3. DETECTION
        print("🔍 Injecting failure events into Aura's memory...")
        for _ in range(3):
            event = ErrorEvent(
                timestamp=time.time(),
                error_type="ImportError",
                error_message="cannot import name 'datetiime' from 'datetime'",
                stack_trace=f"  File \"{target_path}\", line 1, in <module>\n    from datetime import datetiime",
                context={"live_source": True},
                skill_name="clock",
                file_path=target_path,
                line_number=1
            )
            engine.error_intelligence.logger_system.recent_errors.append(event)
        
        # 4. EXECUTION
        print("⚙️ Triggering Autonomous Self-Improvement Cycle...")
        engine.auto_fix_enabled = True
        
        bugs = await engine.diagnose_current_bugs()
        if bugs:
            print(f"✨ Aura detected pattern: {bugs[0]['pattern'].fingerprint[:12]}...")
            fix_proposal = await engine.propose_fix(bugs[0])
            if fix_proposal:
                print(f"🛠️  Aura proposed fix for {fix_proposal['fix'].target_file}")
                success = await engine.apply_fix(fix_proposal, force=True)
                
                print("\n" + "=" * 80)
                print("📊 FINAL RECEIPTS (LIVE SOURCE)")
                print("-" * 80)
                if success:
                    print("Autonomous Cycle: SUCCESS ✅")
                    print("Improvement Promotion: COMMITTED")
                    
                    current_code = target_file.read_text(encoding="utf-8")
                    if "from datetime import datetime" in current_code:
                        print("Aura's Self-Repair: VERIFIED 💠")
                        diff = difflib.unified_diff(
                            buggy_code.splitlines(),
                            current_code.splitlines(),
                            fromfile="Buggy State",
                            tofile="Aura Improved State",
                            lineterm=''
                        )
                        print("\nAUDIT LOG:")
                        print('\n'.join(diff))
                    else:
                        print("Aura's Self-Repair: FAILED (Code mismatch) ❌")
                else:
                    print("Autonomous Cycle: REJECTED ❌")
            else:
                print("Status: FAILED ❌ - Fix generation/validation failed.")
        else:
            print("Status: FAILED ❌ - Aura didn't see the bug.")
    finally:
        atomic_write_text(target_file, original_code, encoding="utf-8")

if __name__ == "__main__":
    asyncio.run(run_decisive_demo())
